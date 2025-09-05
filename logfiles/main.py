# -*- coding: utf-8 -*-
"""
Created on Thu May  1 14:45:45 2025
@author: SeyedHamed.Meshkin
"""
import os
import umap
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from antiberty import AntiBERTyRunner
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import accuracy_score, confusion_matrix
from Immunogenicity import ImmunogenicityPredictor, Hierarchical_Clustering, Single_NN
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score




parser = argparse.ArgumentParser(description='No description')
parser.add_argument('--output', dest='output_file',default = 2, type=int, help='Output file name')
parser.add_argument('--input', dest='input_file',default = 'ver1_1.0%', type=str, help='input file name, Please put underline and percentage sign')
parser.add_argument('--reduction', dest='reduction',default = 'PCA', type=str, help='UMAP, LDA and PCA dimensionality reduction method ')
parser.add_argument('--trained', dest='trained',default = False, type=bool, help='Using my trained AntiBERTy or not ')

args = parser.parse_args()
output_file = args.output_file
input_file = args.input_file
ARRAY_TASK_ID = output_file
reduction = args.reduction
trained_embedding = args.trained

antibody_type = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    DATA_PATH = './data/'
    FILES = {
        "Train": "train_"+input_file+".csv",
        "Test":  "test_" +input_file+".csv"
    }
    File_name = FILES['Train'].rsplit('.', 1)[0].split('_', 1)[1]

    # Model execution
    Model = ImmunogenicityPredictor(data_path=DATA_PATH, file_names=FILES,ARRAY_TASK_ID=ARRAY_TASK_ID,  n_components=2, Split_datasets=True)
    HierC = Hierarchical_Clustering()

    # Step data preparation
    Threshold = float(input_file.split('_')[1].split('%')[0]) # Use parser to modify ADA version and the favorable threshold
    version   = input_file.split('_')[0]
    print(f"\nThreshold is {Threshold}")
    print(f"ADA version is: {version} \n")
    if version=='ver1':
        ADA_version = 'ADA_summary_version_1.csv'
    elif version=='ver2':
        ADA_version = 'ADA_summary_version_2.csv'

    Model.Data_Labeling_Prepration(threshold=Threshold, ADA_version=ADA_version)

    # Step0 Data Loading

    print("\nLoading data...")
    Model.load_data()


    if trained_embedding:
        # Step1 Data Embedding
        print("Embedding training sequences...")
        embeddings_train = Model.embed_sequences_trained(Model.train_dataset)

        print("Embedding testing sequences...")
        embeddings_test = Model.embed_sequences_trained(Model.test_dataset)
    else:
        # Step1 Data Embedding
        print("Embedding training sequences...")
        embeddings_train = Model.embed_sequences(Model.train_dataset)

        print("Embedding testing sequences...")
        embeddings_test = Model.embed_sequences(Model.test_dataset)



    print("Extracting 1024-dim vectors...\n")
    train_vecs = Model.extract_1024_vector(embeddings_train)
    test_vecs = Model.extract_1024_vector(embeddings_test)

    # Step2 Data PCA
    y_train = Model.train_dataset['Immun_label'].values
    y_test =  Model.test_dataset['Immun_label'].values
    if reduction=='PCA':
        print("Applying PCA...")
        X_train, X_test = Model.run_pca(train_vecs, test_vecs)
    elif reduction=='LDA':
        print("Applying LDA...")
        X_train, X_test = Model.run_lda(train_vecs,y_train, test_vecs)
    elif reduction=='UMAP':
        print("Applying uMAP...")
        X_train, X_test  = Model.run_umap(train_vecs, test_vecs, N_neighbors=5, Min_dist=1.0, Metric='cosine', Random_state=1)

    # 3rd element (index 2) from antibody_type
    if antibody_type==True:
        X_test = np.column_stack((X_test, np.full(X_test.shape[0], Model.test_dataset['antibody_type'] )))
        X_train = np.column_stack((X_train, np.full(X_train.shape[0], Model.train_dataset['antibody_type'] )))


    All_data_1024 = np.vstack([train_vecs, test_vecs])
    All_data_redu = np.vstack([X_train, X_test])
    All_labels = np.concatenate([y_train, y_test])
    All_ADA = pd.concat([Model.train_dataset,Model.test_dataset]).reset_index(drop=True)

    # Calculate and Plot Hierarchical Clustering
#    File_name = File_name+'_1024_'
    All_data_redu_unsupervised = Model.run_pca_unsupervised(train_vecs, test_vecs, n_components=2)
    HierC.plot_class_oriented_dendrogram(
                                    X=All_data_redu_unsupervised,
                                    y=All_labels,
                                    method='ward', # ward, single, complete, average, median, weighted, centroid
                                    metric='euclidean', # euclidean,  correlation, hamming, cosine, cityblock, chebyshev, jaccard
                                    target_class=1,
                                    push='right',
                                    figsize=(18, 6),
                                    leaf_font_size=7,
                                    show_strip=True,
                                    File_name= File_name+'test3'
                                )


    HierC.save_cluster_2_csv(All_ADA,All_labels,HierC.dendro)


    exec(open("cluster.py").read())
    exec(open("Benchmark_evaluation.py").read())
    print("\033[91m\nAverage distance calculation:\033[0m")
    Averages_Distance = HierC.average_cross_distance(X=All_data_redu_unsupervised,y=All_labels)
    for key, value in Averages_Distance.items(): print(f"{key}: {value}")


    print(f"\033[91m\nPerforming Neural Network on KFold {ARRAY_TASK_ID} \033[0m")

    # Split and convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32,device=device)
    X_test   = torch.tensor(X_test, dtype=torch.float32,device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32,device=device)
    y_test   = torch.tensor(y_test, dtype=torch.float32,device=device)

    # Wrap in DataLoader


    batch_size = 10000
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model_nn = Single_NN(input_size=X_train.shape[1],Layer_Size=64).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=1e-3)

    # Training loop
    epochs = 40000
    for epoch in range(epochs):
        model_nn.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model_nn(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model_nn.eval()
        if epoch % 100 == 0:
            with torch.no_grad():
                X_test_tensor = X_test.float()  # Ensure float type if not already
                y_test_tensor = y_test.float()  # Ensure float type if not already
                yb_tensor = yb.float()
                preds_val = model_nn(xb).squeeze()
                preds_bin = (preds_val >   0.5).float()

                acc = accuracy_score(yb_tensor.cpu(), preds_bin.cpu())
                print(f"Epoch {epoch+1}/{epochs}, Val Accuracy: {acc:.4f}")




    preds_val = model_nn(X_test_tensor).squeeze()
    preds_bin = (preds_val >   0.5).float()
    Y_True    = y_test.cpu()
    y_Pred = preds_bin.cpu()
    cm = confusion_matrix(Y_True,y_Pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(Y_True, y_Pred)
    precision = precision_score(Y_True, y_Pred)
    f1 = f1_score(Y_True, y_Pred)
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # Optional: print or log individual results
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    correct_mask = (y_Pred - Y_True)==0
    Wrong = Model.test_dataset.iloc[~correct_mask.numpy()]
    print(Wrong)

    # Create DataFrame
    df = pd.DataFrame({
        'Y_true': Y_True,
        'Y_pred': y_Pred
    })

    # Save to CSV
    df.to_csv('output/res_'+ str(output_file) + '.dat', index=False,header=False)
