# -*- coding: utf-8 -*-
"""
Created on Thu May  1 14:45:45 2025
@author: SeyedHamed.Meshkin
"""
import os
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from autogluon.tabular import TabularPredictor
#from antiberty import AntiBERTyRunner
from Immunogenicity import FeedforwardNN
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from Immunogenicity import ImmunogenicityPredictor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import ipdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    DATA_PATH = '../AntiBERTy-main_updated_dataset/data/'
    FILES = {
        "Train": "train_ver1_2.0%.csv",
        "Test": "test_ver1_2.0%.csv"
    }

    # Model execution
    Model = ImmunogenicityPredictor(data_path=DATA_PATH, file_names=FILES,ARRAY_TASK_ID=2)

    # Step0 Data Loading
    print("Loading data...")
    Model.load_data()

    # Step1 Data Embedding
    print("Embedding training sequences...")
    embeddings_train = Model.embed_sequences_trained(Model.train_dataset)

    print("Embedding testing sequences...")
    embeddings_test = Model.embed_sequences_trained(Model.test_dataset)

    print("Extracting 1024-dim vectors...")
    train_vecs = Model.extract_1024_vector(embeddings_train)
    test_vecs = Model.extract_1024_vector(embeddings_test)

    # Step2 Data PCA
    print("Applying PCA...")
    X_train, X_test = Model.run_pca(train_vecs, test_vecs)
    y_train = Model.train_dataset['Immun_label']
    y_test =  Model.test_dataset['Immun_label']

    # Step3 Data MLs
    print("Training and evaluating models...")
    Model.train_evaluate_model(X_train, y_train, X_test, y_test)


    data_train = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(Model.n_components)])
    data_train['Label'] = y_train

    data_test = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(Model.n_components)])
    data_test['Label'] = y_test


# Initialize a dictionary to accumulate metric values
metric_sums = defaultdict(list)
metrics = ['accuracy', 'balanced_accuracy', 'mcc', 'roc_auc', 'f1', 'precision', 'recall']

# Loop over all predictors and collect metrics
for predictor in Model.predictors:
    results = predictor.evaluate(data_test, silent=True)
    for metric in metrics:
        if metric in results:
            metric_sums[metric].append(results[metric])

# Compute average for each metric
print("\nAverage Evaluation Metrics across all folds:")
for metric in metrics:
    if metric in metric_sums:
        avg = np.mean(metric_sums[metric])
        print(f"{metric:>20}: {avg:.4f}")



if False:
    # List all AutoGluon save folders
    folders = [f for f in os.listdir("AutogluonModels") if os.path.isdir(os.path.join("AutogluonModels", f))]
    latest_folder = sorted(folders)[-1]
    # Load it
    predictors = TabularPredictor.load(os.path.join("AutogluonModels", latest_folder))

exec(open('plot.ROC.py').read())



#


