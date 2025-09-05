import os
import pandas as pd
import numpy as np
import sys
import umap
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from autogluon.common import space
from antiberty import AntiBERTyRunner
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from transformers import BertForMaskedLM, PreTrainedTokenizerFast
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import umap

class ImmunogenicityPredictor:
    def __init__(self, data_path, file_names,ARRAY_TASK_ID, n_components=2, Split_datasets=True):
        self.data_path = data_path
        self.file_names = file_names
        self.n_components = n_components
        self.antiberty = AntiBERTyRunner()
        self.pca = PCA(n_components=self.n_components)
        self.lda = LinearDiscriminantAnalysis(n_components=1)
        self.predictors = []  # Add this at the beginning of the method
        self.Split_datasets = Split_datasets
        self.ARRAY_TASK_ID = ARRAY_TASK_ID
#        self.model = BertForMaskedLM.from_pretrained("../Tuned_BERT/trained_antiberty")
#        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("../Tuned_BERT/trained_antiberty")

    def Data_Labeling_Prepration(self, threshold = 2.0, ADA_version = 'ADA_summary_version_1.csv'):

        data_path = ""
        Antibod_updated = pd.read_csv(os.path.join("./data/" ,ADA_version), low_memory= False , sep= ',' )
        Antibod_train   = pd.read_csv(os.path.join("Wang/", 'train_2%.csv'), low_memory=False, sep=',')
        Antibod_test    = pd.read_csv(os.path.join("Wang/", 'test_2%.csv'), low_memory=False, sep=',')


        Antibod_updated.rename(columns={'Antibody drug name': 'Therapeutic_antibody'}, inplace=True)

        Antibod_updated.loc[Antibod_updated['Type of antibodies'].str.strip().str.lower() == 'human', 'antibody_type'] = 0
        Antibod_updated.loc[Antibod_updated['Type of antibodies'].str.strip().str.lower() == 'humanized', 'antibody_type'] = 0
        Antibod_updated.loc[Antibod_updated['Type of antibodies'].str.strip().str.lower() == 'chimeric', 'antibody_type'] = 0
        Antibod_updated.loc[Antibod_updated['Type of antibodies'].str.strip().str.lower() == 'mouse', 'antibody_type'] = 1


        Antibod_updated.set_index('Therapeutic_antibody')['Type of antibodies']
        ###
        Antibod_old = pd.concat([Antibod_train,Antibod_test])

        Antibod_old["Therapeutic_antibody"] = Antibod_old["Therapeutic_antibody"].str.strip().str.lower()
        Antibod_updated["Therapeutic_antibody"] = Antibod_updated["Therapeutic_antibody"].str.strip().str.lower()

        #merged_df = pd.merge(Antibod_old, Antibod_updated, on='Therapeutic_antibody', how='inner')
        #merged_df.to_csv('shared_dataset.csv')

        # Names only in Antibod_old
        only_in_Antibod_old = Antibod_old[~Antibod_old['Therapeutic_antibody'].isin(Antibod_updated['Therapeutic_antibody'])]
        print("only in Wang's Antibod")
        print(only_in_Antibod_old['Therapeutic_antibody'])
        print("##############################################################################################################################")
        # Names only in Antibod_updated
        only_in_Antibod_updated = Antibod_updated[~Antibod_updated['Therapeutic_antibody'].isin(Antibod_old['Therapeutic_antibody'])]
        print("only in Ji Young's Antibod")
        print(only_in_Antibod_updated['Therapeutic_antibody'])
        print("##############################################################################################################################")



        print("Generating test Dataset")
        test = pd.DataFrame()
        test["N0"] =  Antibod_test["N0"]
        test['Therapeutic_antibody'] = Antibod_test["Therapeutic_antibody"]
        test['Heavy_Chain'] = Antibod_test["Heavy_Chain"]
        test['Light_Chain'] = Antibod_test["Light_Chain"]
        test["Therapeutic_antibody"] = test["Therapeutic_antibody"].str.strip().str.lower()
        Immun_value_map = Antibod_updated.set_index('Therapeutic_antibody')['ADA (%) we found (study reports & FDA labels)']
        Immun_type_map = Antibod_updated.set_index('Therapeutic_antibody')['antibody_type']
        test['Immun_value'] = test['Therapeutic_antibody'].map(Immun_value_map)
        test['antibody_type'] = test['Therapeutic_antibody'].map(Immun_type_map)
        test['Immun_label'] = np.where(test['Immun_value'] < threshold, 0, 1)
        #test['Immun_label'] = np.where(test['Immun_value'] < 2, 0,    np.where(test['Immun_value'] <= 4, 1, 2))
        test = test.dropna(axis=0)

        test.to_csv('data/test_ver1_' + str(threshold) + '%.csv', index=False)
        print("##############################################################################################################################")

        print("Generating Train Dataset")
        train = pd.DataFrame()
        train["N0"] = Antibod_train["N0"]
        train['Therapeutic_antibody'] = Antibod_train["Therapeutic_antibody"]
        train["Therapeutic_antibody"] = train["Therapeutic_antibody"].str.strip().str.lower()
        if (only_in_Antibod_updated.shape[0] > 0):    train = pd.concat([train, only_in_Antibod_updated[['Therapeutic_antibody']]], axis=0)
        train = train.reset_index(drop=True)
        train['N0'] = train.index
        heavy_map = Antibod_old.set_index('Therapeutic_antibody')['Heavy_Chain']
        train['Heavy_Chain'] = train['Therapeutic_antibody'].map(heavy_map)
        light_map = Antibod_old.set_index('Therapeutic_antibody')['Light_Chain']
        train['Light_Chain'] = train['Therapeutic_antibody'].map(light_map)
        Immun_value_map = Antibod_updated.set_index('Therapeutic_antibody')['ADA (%) we found (study reports & FDA labels)']
        Immun_type_map = Antibod_updated.set_index('Therapeutic_antibody')['antibody_type']
        train['Immun_value'] = train['Therapeutic_antibody'].map(Immun_value_map)
        train['antibody_type'] = train['Therapeutic_antibody'].map(Immun_type_map)
        train['Immun_label'] = np.where(train['Immun_value'] < threshold, 0, 1)
        #train['Immun_label'] = np.where(train['Immun_value'] < 2, 0,    np.where(train['Immun_value'] <= 4, 1, 2))
        train.loc[train['Therapeutic_antibody'] == 'tibulizumab', 'Heavy_Chain'] = "QVQLVQSGAEVKKPGSSVKVSCKASGYSFTDYHIHWVRQAPGQGLEWMGVINPMYGTTDYNQRFKGRVTITADESTSTAYMELSSLRSEDTAVYYCARYDYFTGTGVYWGQGTLVTVSS"
        train.loc[train['Therapeutic_antibody'] == 'tibulizumab', 'Light_Chain'] = "DIVMTQTPLSLSVTPGQPASISCRSSRSLVHSRGNTYLHWYLQKPGQSPQLLIYKVSNRFIGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCSQSTHLPFTFGQGTKLEIK"

        train.loc[train['Therapeutic_antibody'] == 'pinatuzumab', 'Heavy_Chain'] = "EVQLVESGGGLVQPGGSLRLSCAASGYEFSRSWMNWVRQAPGKGLEWVGRIYPGDGDTQYSGKFKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARDGSSWDWYFDVWGQGTLVTVSS"
        train.loc[train['Therapeutic_antibody'] == 'pinatuzumab', 'Light_Chain'] = "DIQMTQSPSSLSASVGDRVTITCRSSQSIVHSVGNTFLEWYQQKPGKAPKLLIYKVSNRFSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCFQGSQFPYTFGQGTKVEIK"

        train.loc[train['Therapeutic_antibody'] == 'omalizumab', 'Heavy_Chain'] = "EVQLVESGGGLVQPGGSLRLSCAVSGYSITSGYSWNWIRQAPGKGLEWVASIKYSGETKYNPSVKGRITISRDDSKNTFYLQMNSLRAEDTAVYYCARGSHYFGHWHFAVWGQGTLVTVSS"
        train.loc[train['Therapeutic_antibody'] == 'omalizumab', 'Light_Chain'] = "DIQLTQSPSSLSASVGDRVTITCRASKPVDGEGDSYLNWYQQKPGKAPKLLIYAASYLESGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSHEDPYTFGQGTKVEIK"



        train = train.dropna(axis=0)
        train.to_csv('data/train_ver1_' + str(threshold) + '%.csv', index=False)

    def load_data(self):
        self.datasets = {
            key: pd.read_csv(os.path.join(self.data_path, fname), low_memory=False, sep=',')
            for key, fname in self.file_names.items()
        }
        combined_df = pd.concat([self.datasets['Train'], self.datasets['Test']], axis=0).reset_index(drop=True)
#        train_df, test_df = train_test_split(combined_df, stratify=combined_df['Immun_label'], test_size=0.2, random_state=42, shuffle=True)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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


    def embed_sequences_trained(self, dataset):
        sequences = dataset[['Heavy_Chain', 'Light_Chain']]
        inputs = [self.tokenizer(list(sequences.iloc[i]), return_tensors="pt",padding=True, truncation=True,return_attention_mask=True) for i in range(len(sequences))]

        with torch.no_grad():
            outputs = [self.model(**inputs[i], output_hidden_states=True) for i in range(len(inputs))]


        embeddings = [torch.mean(torch.stack(outputs[i].hidden_states[-1:]), dim=0) for i in range(len(inputs))]
        return embeddings

    def embed_sequences(self, dataset):
        sequences = dataset[['Heavy_Chain', 'Light_Chain']]
        embeddings = [self.antiberty.embed(sequences.iloc[i]) for i in range(len(sequences))]
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

    def run_pca_unsupervised(self, train_vecs, test_vecs, n_components=2):
        All_data_1024 = np.vstack([train_vecs, test_vecs])
        return PCA(n_components=n_components).fit(All_data_1024).transform(All_data_1024)

    def run_lda(self,train_vecs,y_train,test_vecs):
#        Zs = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=0).fit_transform(train_vecs, y)
        self.lda.fit_transform(train_vecs,y_train)
        X_train_pca = PCA(n_components=1).fit(train_vecs).transform(train_vecs)
        X_test_pca  = PCA(n_components=1).fit(train_vecs).transform(test_vecs)
        X_train_lda = self.lda.transform(train_vecs)
        X_test_lda = self.lda.transform(test_vecs)
        return np.hstack([X_train_pca, X_train_lda]), np.hstack([X_test_pca, X_test_lda])

    def run_umap(self, train_vecs, test_vecs,N_neighbors=15,Min_dist=0.1,Metric='cosine', Random_state=0):
        mapper  = umap.UMAP(n_neighbors=N_neighbors, min_dist=Min_dist, metric=Metric, random_state=Random_state).fit(train_vecs)
        return mapper.transform(train_vecs),mapper.transform(test_vecs)



    def train_evaluate_model(self, X_train, y_train, X_test, y_test):
        data_train = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])])
        data_train['Label'] = y_train.values

        data_test = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])])
        data_test['Label'] = y_test.values

        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_pos_weight =  num_pos / num_neg

        model_list = [
            'GBM', 'CAT', 'XGB', 'RF', 'XT', 'KNN', 'NN_TORCH', 'FASTAI']
#           LightGBM, CatBoost, XGBoost, Random Forest, Extra Trees, KNN, Neural Net torch, Neural Net (FastAI)
#        hyperparameters = {model: {} for model in model_list}
#        nn_options = {  # specifies non-default hyperparameter values for neural network models
#            'num_epochs': 1000,  # number of training epochs (controls training time of NN models)
#            'learning_rate': space.Real(1e-4, 1e-2, default=5e-4, log=True),  # learning rate used in training (real-valued hyperparameter searched on log-scale)
#            'activation': space.Categorical('relu', 'softrelu', 'tanh'),  # activation function used in NN (categorical hyperparameter, default = first entry)
#            'dropout_prob': space.Real(0.0, 0.5, default=0.1),  # dropout probability (real-valued hyperparameter)
#        }
        nn_options = {
            'num_epochs': 20000,  # Allow more training cycles
            'learning_rate': 1e-3,  # Start with 1e-3 or tune it
            'activation': 'relu',  # Try 'tanh' if 'relu' isn't helping
            'dropout_prob': 0.2,  # Helps regularize if overfitting
            'weight_decay': 1e-5,  # L2 regularization
        }

        gbm_options = {  # specifies non-default hyperparameter values for lightGBM gradient boosted trees
            'num_boost_round': 100,  # number of boosting rounds (controls training time of GBM models)
            'num_leaves': space.Int(lower=26, upper=66, default=36),  # number of leaves in trees (integer hyperparameter)
        }

        hyperparameters = {
                'CAT': {
                    'scale_pos_weight': scale_pos_weight
                       },
                'XGB': {
                    'scale_pos_weight': scale_pos_weight  # Manual ratio for class balancing
                       },
                'NN_TORCH': nn_options,
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

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.scores = []
#        ipdb.set_trace()
        X = data_train.drop(columns='Label').values
        y = data_train['Label'].values

        for fold, (train_idx, val_idx) in enumerate(kf.split(X,y), 1):
            train_fold = data_train.iloc[train_idx]
            valid_fold = data_train.iloc[val_idx]
            train_fold = data_train
            predictor = TabularPredictor(label='Label', problem_type='binary', verbosity=1, eval_metric='log_loss' ).fit(
                train_data=train_fold,
#                tuning_data=valid_fold,
                hyperparameters=hyperparameters,
                presets='best_quality',#'best_quality', #high_quality
#                refit_full=True,
#                feature_prune=True,
                num_gpus=1,
                ag_args_fit={"random_seed": 42},
                num_bag_folds=5
            )#.fit_weighted_ensemble()

#            ipdb.set_trace()
            lb = predictor.leaderboard(silent=True)
            with open('leaderboard_output.txt', 'a') as f:
                f.write(lb.to_string(index=False) + '\n')

            self.predictors.append(predictor)
#            ipdb.set_trace()
            perf = predictor.evaluate(data_test, silent=True)
            print(f"Fold {fold} Accuracy: {perf['accuracy']:.4f}")
            self.scores.append(perf['accuracy'])
            print("H########H")
             # Evaluate log loss (aka performance loss)
            print("Log Loss:", perf['log_loss'])
            sys.stdout.flush()
            break
        print(f"\nAverage 5-Fold Accuracy: {np.mean(self.scores):.4f}")


class Hierarchical_Clustering:
    def __init__(self):
        self.dendro = []
        self.Z = []
        self.fig = []
        self.ax = []

    def _to_numpy(self,x):
        # Handles numpy arrays, lists, and torch tensors (CPU/CUDA)
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def reorder_linkage_for_class(self, Z, y, target_class=1, push='right'):
        """
        Flip children at each merge so the subtree with more target_class goes to
        the requested side ('right' or 'left'). Returns a reordered copy of Z.
        """

        Z = Z.copy()
        n = Z.shape[0] + 1

        total = np.zeros(n + Z.shape[0], dtype=int)
        hits  = np.zeros(n + Z.shape[0], dtype=int)

        total[:n] = 1
        hits[:n]  = (y == target_class).astype(int)

        for i in range(Z.shape[0]):
            a, b = int(Z[i, 0]), int(Z[i, 1])
            node = n + i
            total[node] = total[a] + total[b]
            hits[node]  = hits[a] + hits[b]

            # Decide which child should be on the RIGHT (or LEFT)
            fa = hits[a] / total[a]
            fb = hits[b] / total[b]
            # who is "preferred" (has more target class)?
            prefer_a = fa > fb or (fa == fb and hits[a] > hits[b])
            # swap if our preferred side doesn't match the requested push
            if (push == 'right' and prefer_a) or (push == 'left' and not prefer_a):
                Z[i, 0], Z[i, 1] = Z[i, 1], Z[i, 0]

        return Z


    def color_dendrogram_by_side(self, Z, ax=None, left_color='tab:blue', right_color='tab:red', **kwargs):
        """
        Color the dendrogram so that everything under the root's LEFT child is
        left_color and everything under the RIGHT child is right_color.
        Colors both links (nodes) and leaf tick labels.
        """
        n = Z.shape[0] + 1
        ax = plt.gca() if ax is None else ax

        # --- build child map for quick traversal
        children = {n+i: (int(Z[i,0]), int(Z[i,1])) for i in range(n-1)}

        def leaves_under(node):
            out, stack = [], [node]
            while stack:
                k = stack.pop()
                if k < n: out.append(k)
                else:     stack.extend(children[k])
            return out

        # root split
        left_root, right_root = int(Z[-1,0]), int(Z[-1,1])
        left_leaves  = set(leaves_under(left_root))
        right_leaves = set(leaves_under(right_root))

        # mark membership (for every node)
        in_left  = np.zeros(n + (n-1), dtype=bool)
        in_right = np.zeros(n + (n-1), dtype=bool)

        for i in range(n):
            in_left[i]  = i in left_leaves
            in_right[i] = i in right_leaves
        for i in range(n-1):
            node = n + i
            a, b = children[node]
            in_left[node]  = in_left[a]  or in_left[b]
            in_right[node] = in_right[a] or in_right[b]

        def link_color_func(k):
            # color internal links by side; root (spans both) -> gray
            if in_left[k] and not in_right[k]:
                return left_color
            if in_right[k] and not in_left[k]:
                return right_color
            return '0.5'  # root

        # draw dendrogram with our link colors
        dendro = dendrogram(Z, ax=ax, link_color_func=link_color_func,
                            color_threshold=0, **kwargs)

        # color leaf tick labels by side
        for txt, leaf_id in zip(ax.get_xmajorticklabels(), dendro['leaves']):
            txt.set_color(right_color if leaf_id in right_leaves else left_color)

        return dendro, left_leaves, right_leaves


    def plot_class_oriented_dendrogram(self,
        X, y, method='ward', metric='euclidean', target_class=1, push='right',
        scale=True, optimal_ordering=True, figsize=(14, 6), leaf_font_size=8,
        show_strip=True, title=None, xlabel='Antibody Index', ylabel='Distance',File_name='File_name'):
        """
        Draw a dendrogram with subtree orientation chosen to push `target_class`
        to the specified side ('right' or 'left'). Colors leaf tick labels and
        adds an optional class strip beneath the leaves.
        """
        X = self._to_numpy(X)
        y = self._to_numpy(y).ravel()
        assert X.shape[0] == y.shape[0], "X rows must align with y length."

        if scale:
            X = StandardScaler().fit_transform(X)

        # ward only supports Euclidean in SciPy; ignore metric if ward is used
        if method == 'ward':
            Z = linkage(X, method=method, optimal_ordering=optimal_ordering)
        else:
            Z = linkage(X, method=method, metric=metric, optimal_ordering=optimal_ordering)

        Z_re = self.reorder_linkage_for_class(Z, y, target_class=target_class, push=push)



        fig, ax = plt.subplots(figsize=figsize)
#        dendro = dendrogram(
#            Z_re, ax=ax,
#            color_threshold=0,
#            leaf_font_size=leaf_font_size,
#            no_labels=False,
#            leaf_rotation=90,
#        )
        dendro, left_set, right_set = self.color_dendrogram_by_side(Z_re, ax=ax, leaf_font_size=8)
        ax.set_title('Left vs Right colored dendrogram')
        ax.set_xlabel('Samples'); ax.set_ylabel('Distance')
#        side_vec = np.array([1 if i in right_set else 0 for i in dendro['leaves']])[None, :]
#        ax2 = ax.inset_axes([0, -0.15, 1, 0.04])
#        ax2.imshow(side_vec, aspect='auto', cmap=ListedColormap(['tab:blue', 'tab:red']))
#        ax2.set_xticks([]); ax2.set_yticks([])

        leaf_order = dendro['leaves']
        tick_texts = ax.get_xmajorticklabels()
        for txt, idx in zip(tick_texts, leaf_order):
            txt.set_color('red' if y[idx] == target_class else 'blue')

        ax.set_title(title or f'Hierarchical Clustering (class {target_class} → {push})')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if show_strip:
            ax2 = ax.inset_axes([0, -0.15, 1, 0.04])  # x,y,w,h (axes fraction)
            strip = (y[leaf_order] == target_class).astype(int)[None, :]
            ax2.imshow(strip, aspect='auto', cmap=ListedColormap(['blue', 'red']))
            ax2.set_xticks([])
            ax2.set_yticks([])

        fig.tight_layout()

        self.dendro = dendro
        self.Z = Z_re
        self.fig = fig
        self.ax = ax

        N_predict_class_1 = np.array([tmp=='tab:blue' for tmp in dendro['leaves_color_list']]).sum()
        N_predict_class_2 = np.array([tmp=='tab:red' for tmp in dendro['leaves_color_list']]).sum()
        cluster_order_class =  y[leaf_order]

        left_predict  = cluster_order_class[0:N_predict_class_1]
        right_predict = cluster_order_class[N_predict_class_1:]

        FP = (left_predict==1).sum().item()
        TP = (left_predict==0).sum().item()

        FN = (right_predict==0).sum().item()
        TN = (right_predict==1).sum().item()

        true_percentage_class_1   = np.round(100 * TP / (TP + FP)).item()
        false_percentage_class_1  = np.round(100-100 * TP / (TP + FP)).item()
        true_percentage_class_2  = np.round(100 * TN / (TN + FN)).item()
        false_percentage_class_2 = np.round(100-100 * TN / (TN + FN)).item()

        # Build multi-line text
        stats_text = (
            f"Percentage cluster blue, <{File_name.split('_')[1]} = {true_percentage_class_1}%  -> {TP}  out of {TP + FP}  \n"
            f"Percentage cluster blue, >{File_name.split('_')[1]} = {false_percentage_class_1}%  -> {FP}  out of {TP + FP} \n\n"

            f"Percentage cluster red, >{File_name.split('_')[1]} = {true_percentage_class_2}%  -> {TN}  out of {TN + FN}  \n"
            f"Percentage cluster red, <{File_name.split('_')[1]} = {false_percentage_class_2}%  -> {FN}  out of {TN + FN}"
        )

        # Place the text inside the plot
        ax.text(
            0.15, 0.90, stats_text,
            transform=ax.transAxes,  # relative coordinates (0–1)
            fontsize=9,
            va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray')  # optional background box
        )
        plt.tight_layout()
        plt.savefig('plot/Hierarchical_2cluster_'+File_name+'.png')

    def save_cluster_2_csv(self,All_ADA,All_labels,dendro):
        leaf_order = dendro['leaves']
        N_predict_class_1 = np.array([tmp=='tab:blue' for tmp in dendro['leaves_color_list']]).sum()
        N_predict_class_2 = np.array([tmp=='tab:red' for tmp in dendro['leaves_color_list']]).sum()
        cluster_order_class = All_labels[leaf_order]

        left_drugs = All_ADA['Therapeutic_antibody'][leaf_order][0:N_predict_class_1]
        right_drugs = All_ADA['Therapeutic_antibody'][leaf_order][N_predict_class_1:]

        left_score = All_ADA['Immun_value'][leaf_order][0:N_predict_class_1]
        right_score = All_ADA['Immun_value'][leaf_order][N_predict_class_1:]

        left_label = cluster_order_class[0:N_predict_class_1]
        right_label = cluster_order_class[N_predict_class_1:]
        True_label_left = ['red' if tmp==1 else 'blue' for tmp in left_label]
        True_label_right = ['red' if tmp==1 else 'blue' for tmp in right_label]
        Left = pd.DataFrame({'Therapeutic_antibody':left_drugs, 'Immun_value':left_score , 'True_label':True_label_left, 'Cluster_label':len(left_label)*['blue']})
        Right = pd.DataFrame({'Therapeutic_antibody':right_drugs, 'Immun_value':right_score , 'True_label':True_label_right, 'Cluster_label':len(right_label)*['red']})

        All_drug =  pd.concat([Left,Right],axis=0)
        All_drug.to_csv('output/2-Cluster.csv')

    def average_cross_distance(self, X, y, metric='euclidean', standardize=True,sample_pairs=None, random_state=0):
        """
        Mean distance between all red/blue pairs:
            μ_between = (1 / (|A||B|)) * sum_{i in A} sum_{j in B} d(x_i, x_j)
        Also returns within-class means, an Energy-distance style score, and a ratio.

        Parameters
        ----------
        X : (n, d) array
        y : (n,) array of 0/1 (two groups)
        metric : str or callable for distances (e.g., 'euclidean', 'cosine', 'cityblock')
        standardize : bool, z-score features before computing distances
        sample_pairs : int or None. If set, subsamples to ~this many cross pairs
                       for speed (recommended when n0*n1 is huge).
        random_state : int

        Returns
        -------
        dict with keys:
          - mean_between
          - mean_within_0, mean_within_1
          - energy_distance = 2*between - within_0 - within_1
          - ratio = between / mean(within_0, within_1)
          - silhouette (if metric == 'euclidean', else None)
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel().astype(int)
        assert set(np.unique(y)) <= {0,1}, "y must be binary (0/1)."

        if standardize:
            X = StandardScaler().fit_transform(X)

        A = X[y == 1]  # red
        B = X[y == 0]  # blue
        nA, nB = len(A), len(B)
        if nA == 0 or nB == 0:
            raise ValueError("Both groups must be non-empty.")

        # Optional subsampling for speed (roughly target sample_pairs cross distances)
    #    if sample_pairs is not None and nA*nB > sample_pairs:
    #        rng = np.random.default_rng(random_state)
    #        # choose subsample sizes ~ sqrt(sample_pairs)
    #        k = max(2, int(np.sqrt(sample_pairs)))
    #        idxA = rng.choice(nA, size=min(nA, k), replace=False)
    #        idxB = rng.choice(nB, size=min(nB, k), replace=False)
    #        A_sub, B_sub = A[idxA], B[idxB]
    #    else:
        A_sub = A
        B_sub = B


        # Between-group mean distance
        D_ab = cdist(A_sub, B_sub, metric=metric)
        mean_between = float(D_ab.mean())

        # Within-group means (handle singleton cases)
        mean_within_1 = float(pdist(A_sub, metric=metric).mean()) if len(A_sub) > 1 else 0.0
        mean_within_0 = float(pdist(B_sub, metric=metric).mean()) if len(B_sub) > 1 else 0.0

        # Energy-like separation (larger is better). For Euclidean, this matches Energy distance up to constants.
        energy_distance = 2*mean_between - (mean_within_0 + mean_within_1)

        # Scale-free ratio (larger is better)
        denom = (mean_within_0 + mean_within_1) / 2 if (mean_within_0 + mean_within_1) > 0 else np.nan
        ratio = mean_between / denom if denom and not np.isnan(denom) else np.inf

        # Silhouette (only well-defined for Euclidean here)
        sil = silhouette_score(X, y) if metric == 'euclidean' and len(np.unique(y)) == 2 else None

        return {
            "Mean Between": mean_between,
            "Mean Within_0": mean_within_0,
            "Mean Within_1": mean_within_1,
            "Energy Distance": energy_distance,
            "Ratio": ratio,
            "Silhouette": sil,
            "nA": len(A_sub), "nB": len(B_sub)
        }

# Define the model single class
class Single_NN(nn.Module):
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

#Multiple class
class Multiple_NN(nn.Module):
    def __init__(self, input_size=1024, num_classes=3):  # set number of classes
        super().__init__()  #
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(64, num_classes)  # output logits for each class

    def forward(self, x):
        x = F.relu(self.fc1(x))
#        x = self.dropout(x)
#        x = F.relu(self.fc2(x))
        x = self.out(x)  # logits, do NOT apply softmax here
        return x