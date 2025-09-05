import os
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from autogluon.tabular import TabularPredictor
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
#from B_train import AntiBERTy_2
from Antiberty_2 import AntiBERTy_2
import ipdb

from transformers import BertForMaskedLM, PreTrainedTokenizerFast





class ImmunogenicityPredictor:
    def __init__(self, data_path, file_names, ARRAY_TASK_ID, n_components=2):
        self.data_path = data_path
        self.file_names = file_names
        self.n_components = n_components
        self.antiberty = AntiBERTy_2()
        self.pca = PCA(n_components=self.n_components)
        self.predictors = []  # Add this at the beginning of the method
        self.model = BertForMaskedLM.from_pretrained("trained_antiberty")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("trained_antiberty")
        self.Split_datasets = True
        self.ARRAY_TASK_ID = ARRAY_TASK_ID

    def load_data(self):
        self.datasets = {
            key: pd.read_csv(os.path.join(self.data_path, fname), low_memory=False, sep=',')
            for key, fname in self.file_names.items()
        }
    def load_data(self):
        self.datasets = {
            key: pd.read_csv(os.path.join(self.data_path, fname), low_memory=False, sep=',')
            for key, fname in self.file_names.items()
        }
        combined_df = pd.concat([self.datasets['Train'], self.datasets['Test']], axis=0).reset_index(drop=True)
        train_df, test_df = train_test_split(combined_df, stratify=combined_df['Immun_label'], test_size=0.2, random_state=42, shuffle=True)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
        for fold, (train_idx, test_idx) in enumerate(skf.split(combined_df, combined_df['Immun_label'])):

            if fold==self.ARRAY_TASK_ID:
                train_df = combined_df.iloc[train_idx]
                test_df = combined_df.iloc[test_idx]

#        test_df = combined_df.iloc[[self.ARRAY_TASK_ID]]  # double brackets to keep it as a DataFrame
#        train_df = combined_df.drop(self.ARRAY_TASK_ID)
#        min_class_count = temp_test_df['Immun_label'].value_counts().min()
#
#        balanced_test_df = (
#            temp_test_df
#            .groupby('Immun_label', group_keys=False)
#            .apply(lambda x: x.sample(n=min_class_count, random_state=42))
#        )
#
#        test_df = balanced_test_df.reset_index(drop=True)
#        leftover_test_df = pd.concat([temp_test_df, balanced_test_df]).drop_duplicates(keep=False)
#        train_df = pd.concat([temp_train_df, leftover_test_df]).reset_index(drop=True)
#        train_df = train_df.reset_index(drop=True)

        if not self.Split_datasets:
            self.train_dataset = self.datasets['Train']
            self.test_dataset = self.datasets['Test']
        elif self.Split_datasets:
            self.train_dataset = train_df
            self.test_dataset = test_df

    def embed_sequences(self, dataset):
        sequences = dataset[['Heavy_Chain', 'Light_Chain']]
        embeddings = [self.antiberty(list(sequences.iloc[i])) for i in range(len(sequences))]
        return embeddings

    def embed_sequences_trained(self, dataset):
        sequences = dataset[['Heavy_Chain', 'Light_Chain']]
        inputs = [self.tokenizer(list(sequences.iloc[i]), return_tensors="pt",padding=True, truncation=True,return_attention_mask=True) for i in range(len(sequences))]

        with torch.no_grad():
            outputs = [self.model(**inputs[i], output_hidden_states=True) for i in range(len(inputs))]


        embeddings = [torch.mean(torch.stack(outputs[i].hidden_states[-1:]), dim=0) for i in range(len(inputs))]
        return embeddings

    @staticmethod
    def extract_1024_vector(embeddings):
        vectors = []
        for embedding in embeddings:
            heavy_avg = torch.mean(embedding[0], dim=0).cpu()
            light_avg = torch.mean(embedding[1], dim=0).cpu()
            combined = torch.cat([heavy_avg, light_avg], dim=0)
            vectors.append(combined.numpy())
        return np.stack(vectors)

    def run_pca(self, train_vecs, test_vecs):
        self.pca.fit(train_vecs)
        return self.pca.transform(train_vecs), self.pca.transform(test_vecs)

    def train_evaluate_model(self, X_train, y_train, X_test, y_test):
        data_train = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(self.n_components)])
        data_train['Label'] = y_train

        data_test = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(self.n_components)])
        data_test['Label'] = y_test

        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_pos_weight =  num_pos / num_neg

        model_list = [
            'GBM', 'CAT', 'XGB', 'RF', 'XT', 'KNN', 'NN_TORCH', 'FASTAI']
#           LightGBM, CatBoost, XGBoost, Random Forest, Extra Trees, KNN, Neural Net torch, Neural Net (FastAI)
#        hyperparameters = {model: {} for model in model_list}
        hyperparameters = {
                'CAT': {
                    'scale_pos_weight': scale_pos_weight
                       },
                'XGB': {
                    'scale_pos_weight': scale_pos_weight  # Manual ratio for class balancing
                       },
                'NN_TORCH': {},
                'RF': {},
                'XT': {},
                'KNN': {},
                'GBM': {},
                'ENS_WEIGHTED': {},
                'FASTAI': {}
            }

        '''
        'RF', 'XT', 'KNN', 'GBM', 'CAT', 'XGB', 'NN_TORCH', 'LR',
        'FASTAI', 'TRANSF', 'AG_TEXT_NN', 'AG_IMAGE_NN', 'AG_AUTOMM',
        'FT_TRANSFORMER', 'TABPFN', 'TABPFNMIX', 'FASTTEXT', 'ENS_WEIGHTED',
        'SIMPLE_ENS_WEIGHTED', 'IM_RULEFIT', 'IM_GREEDYTREE', 'IM_FIGS', 'IM_HSTREE',
        'IM_BOOSTEDRULES', 'VW', 'DUMMY'
        '''

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(data_train), 1):
            train_fold = data_train.iloc[train_idx]
            predictor = TabularPredictor(label='Label', problem_type='binary', verbosity=1 ).fit(
                train_fold,
                hyperparameters=hyperparameters,
                presets='best_quality', #high_quality
#                refit_full=True,
#                feature_prune=True,
                num_bag_folds=5
            )#.fit_weighted_ensemble()
            self.predictors.append(predictor)

            perf = predictor.evaluate(data_test, silent=True)
            print(f"Fold {fold} Accuracy: {perf['accuracy']:.4f}")
            self.scores.append(perf['accuracy'])

        print(f"\nAverage 5-Fold Accuracy: {np.mean(self.scores):.4f}")


# Define the model
class FeedforwardNN(nn.Module):
    def __init__(self, input_size=1024, Layer_Size=512 , num_classes=1):  # set num_classes > 1 for multiclass
        super().__init__()
        self.fc1 = nn.Linear(input_size, Layer_Size)
#        self.fc2 = nn.Linear(Layer_Size, 32)
        self.out = nn.Linear(Layer_Size, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
#        x = self.dropout(x)
#        x = F.relu(self.fc2(x))
        x = self.out(x)
        return torch.sigmoid(x) if self.out.out_features == 1 else x  # sigmoid for binary, raw logits for multiclass