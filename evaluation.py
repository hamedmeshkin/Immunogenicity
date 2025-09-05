from sklearn.preprocessing import StandardScaler  # For standardizing features by removing the mean and scaling to unit variance
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score  # For evaluating model performance
import matplotlib.pyplot as plt  # For plotting graphs
import os  # For interacting with the operating system
import numpy as np  # For numerical operations
import argparse  # For parsing command line arguments
from collections import Counter  # For counting hashable objects
import pandas as pd  # For data manipulation and analysis
import ipdb  # For the Python debugger
import glob



parser = argparse.ArgumentParser(description='No description')
parser.add_argument('--input', dest='input_file',default = 'output', type=str, help='input folder name')

args = parser.parse_args()
input_file = args.input_file
# Define a function to calculate and return various metrics based on the predictions
def metrics(mainFolder):
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--sampling', dest='sampling', default="under", type=str, help='Input sampling mode, over / under')
    args = parser.parse_args()
    sampling = args.sampling

    # Initialize a counter for wrong labels
    WrongLable = 0

    Range = range(1, 100)
    FILEname = "res_*"

    folder = mainFolder+'/'
    Files = glob.glob(os.path.join(folder, FILEname))

    # Initialize lists to store various metrics
    Precision = []
    Sensitivity = []
    Acuracy = []
    F1_score = []
    Specificity = []

    # Loop through each file in the specified range
    for FileLocation in Files:
        try:
#            FileLocation = folder + FILE + str(ii) + ".txt"

            # Skip if the file doesn't exist
            if not os.path.exists(FileLocation):
                continue

            labels = []
            # Read labels from the file
            with open(FileLocation, 'r') as file:
                for line in file:
                    tmp = line.strip().split(' ')
                    if len(tmp) < 2:
                        tmp = line.strip().split(',')
                    labels.append(tmp)

            # Split labels into first and second elements, handling cases with missing second elements
            first_elements = []
            second_elements = []
            for sublist in labels:
                if len(sublist) == 2:
                    first_elements.append(sublist[0])
                    second_elements.append(sublist[1])
                else:
                    first_elements.append(sublist[0])
                    second_elements.append('empty')

            # Determine unique classes and correct predictions not matching any class
            classes = list(set(first_elements))
            for idx, pred in enumerate(second_elements):
                if pred not in classes:
                    if first_elements[idx] == classes[0]:
                        second_elements[idx] = classes[1]
                    elif first_elements[idx] == classes[1]:
                        second_elements[idx] = classes[0]

            # Skip file if all predictions are the same or if there are no predictions
            if len(second_elements) == 0 or (second_elements.count(second_elements[0]) == len(second_elements)):
                WrongLable += 1
                continue

            # Convert labels to binary format based on specific criteria
            y_true = []
            y_pred = []
            for ai,bi in zip(first_elements,second_elements):
                    if (ai == '1.0' or ai == '1' or ai == 'Yes' or ai == 'True'):
                        y_true.append(1)
                    else:
                        y_true.append(0)

                    if (bi == '1.0' or bi== '1' or bi == 'Yes' or bi == 'True'):
                        y_pred.append(1)
                    else:
                        y_pred.append(0)

            # Calculate confusion matrix and normalize it
            conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
            conf_matrix = conf_matrix * 100 / np.sum(conf_matrix)

            # Extract true negatives, false positives, false negatives, and true positives
            tn, fp, fn, tp = conf_matrix.ravel()

            # Calculate sensitivity, specificity, precision, accuracy, and F1 score
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            PRECISION = 100.0 * precision_score(y_true, y_pred)
            SENSITIVITY = 100.0 * sensitivity
            ACURACY = 100.0 * accuracy_score(y_true, y_pred)
            F1_SCORE = 100.0 * f1_score(y_true, y_pred)
            SPECIFICITY = 100.0 * specificity

            # Skip if precision is zero
            if PRECISION == 0.0:
                continue

            # Append calculated metrics to their respective lists
            Precision.append(PRECISION)
            Sensitivity.append(SENSITIVITY)
            Acuracy.append(ACURACY)
            F1_score.append(F1_SCORE)
            Specificity.append(SPECIFICITY)
        except Exception as e:
            print("Error occurred:", e)

            ipdb.set_trace()  # Trigger debugger on exception

    # Print the number of trials and calculate the mean and standard deviation of each metric
    print('Number of trials is:')
    print(len(Precision))
    preMean = np.mean(Precision) / 100
    preStd = np.std(Precision) / 100
    senMean = np.mean(Sensitivity) / 100
    senStd = np.std(Sensitivity) / 100
    f1Mean = np.mean(F1_score) / 100
    f1Std = np.std(F1_score) / 100
    speMean = np.mean(Specificity) / 100
    speStd = np.std(Specificity) / 100
    aceMean = np.mean(Acuracy) / 100
    aceStd = np.std(Acuracy) / 100

    # Compile metrics into a DataFrame and return
    Values = {'pre': [preMean, preStd], 'sen': [senMean, senStd], 'f1': [f1Mean, f1Std], 'spe': [speMean, speStd], 'Ace': [aceMean, aceStd]}
    return pd.DataFrame(data=Values)

# Define a list of folders containing validation results
Folders = [input_file]
# Initialize lists to store aggregated metrics and their errors
pre = []; spe = []; f1 = []; sen = []
pre_error = []; spe_error = []; f1_error = []; sen_error = []

# Iterate through each folder, calculate metrics, and append results to the lists
for i, folder in enumerate(Folders):
    print(folder)
    measurements = metrics(folder)
    print(measurements)
    pre.append(measurements['pre'][0])
    sen.append(measurements['sen'][0])
    f1.append(measurements['f1'][0])
    spe.append(measurements['spe'][0])
    pre_error.append(measurements['pre'][1])
    sen_error.append(measurements['sen'][1])
    f1_error.append(measurements['f1'][1])
    spe_error.append(measurements['spe'][1])


