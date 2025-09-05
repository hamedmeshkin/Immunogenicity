import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from matplotlib import cm
from matplotlib.colors import ListedColormap
import ipdb
from builtins import str


if False:

    # Step 1: Put PCA results into a DataFrame
    pca_df = pd.DataFrame(X_train)
    # Step 2: Scale PCA features before clustering
    scaled = StandardScaler().fit_transform(pca_df)
    # Step 3: Perform hierarchical clustering
    linkage_matrix = linkage(scaled, method='ward', optimal_ordering=True)
    labels = y_train  # your binary ADA labels (0 or 1)
    # Define a color function for label colors
    # `labels` should be aligned to the order of the dendrogram leaves
    # This maps label (0 or 1) to color
    def label_color_func(label_index):
        return 'red' if labels[label_index] == 1 else 'blue'
    # Get current axes after plotting dendrogram
    plt.figure(figsize=(16, 10))
    ax = plt.gca()
    # Get x-axis tick labels
    labels = np.array(y_train )
    x_labels = ax.get_xmajorticklabels()
    # Dendrogram returns the leaf order in 'leaves' attribute
    # This ensures labels are matched to dendrogram leaves
    # Plot dendrogram with colored leaf labels

    dendro = sch.dendrogram(
        linkage_matrix,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=0,  # disables branch coloring so only leaf labels show color
    )

    labels = np.array(y_train )
    x_labels = ax.get_xmajorticklabels()

    for i, label in enumerate(x_labels):
        leaf_index = dendro['leaves'][i]
        label.set_color(label_color_func(leaf_index))


    # Step 4: Plot the dendrogram
#    labels = np.array(y_train )
#    x_labels = ax.get_xmajorticklabels()
    dendrogram(linkage_matrix, labels=None)  # optionally use: labels=labels
    plt.title('Hierarchical Clustering')
    plt.xlabel('Antibody Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig('plot/Hierarchical_2%')



# #######################################################################################################
if True:


    # ---- Helper Plotting Function ---- #
    def plot_pca_scatter(data, labels, title, split_label=None, highlight_mask=None):
        plt.figure(figsize=(16, 8))
        unique_classes = np.unique(labels)

        for cls in unique_classes:
            idx = labels == cls
            plt.scatter(
                data[idx, 0], data[idx, 1],
                label=f'ADA Class {cls}',
                alpha=0.6
            )

        # Overlay correctly predicted points if provided
        if highlight_mask is not None:
            plt.scatter(
                data[highlight_mask, 0],
                data[highlight_mask, 1],
                s=80,  # Bigger size
                edgecolors='green',  # <-- Changed from black to red
                facecolors='none',
                linewidths=1.2,
                label="Correctly Predicted"
            )



        if True:
            for idx, (x, y) in enumerate(data):
                plt.text(x, (y+0.35), str(idx), fontsize=8, ha='center', va='center')

        x_min, x_max = data[:, 0].min() - 2, data[:, 0].max() + 2
        y_min, y_max = data[:, 1].min() - 2, data[:, 1].max() + 2

        names = Model.test_dataset['Therapeutic_antibody']
        text_x = x_max + 0.5
        text_y_start = y_max
        line_height = (y_max - y_min) / max(30, len(names))  # space between lines

        if split_label is not None:
            plt.title(f"{title} ({split_label})", fontsize=14)
        else:
            plt.title(title, fontsize=14)
        plt.xlabel("PC1", fontsize=12)
        plt.ylabel("PC2", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        address = 'plot/' + title.replace(" ", "_") + '.png'
#        plt.savefig(address)
        plt.savefig(address, format="png", bbox_inches="tight")
        plt.close()

    # ---- Plot Training, Test, Combined ---- #
    plot_pca_scatter(X_train, np.array(y_train), title="2D PCA of Training_"+File_name)
    plot_pca_scatter(X_test, np.array(y_test), title="2D PCA of Test_"+File_name, highlight_mask=correct_mask)
    plot_pca_scatter(All_data_redu_unsupervised, All_labels, title="2D PCA of All Data_2"+File_name)


if False:
    # Get bounds from PCA components
    def Contour(X,Y,named):
        x_min, x_max = xb[:, 0].min() - 2, xb[:, 0].max() + 2
        y_min, y_max = xb[:, 1].min() - 2, xb[:, 1].max() + 2

        # Create grid
        xx, yy = np.meshgrid(np.linspace(x_min.item(), x_max.item(), 1500),
                             np.linspace(y_min.item(), y_max.item(), 1500))

        # Flatten and stack for prediction
        grid_2D = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
        preds_grid = model_nn(grid_2D).squeeze()
        preds_grid_bin = (preds_grid >   0.5).float()

        # Reshape predictions to match grid
        Z = preds_grid_bin.detach().cpu().numpy().reshape(xx.shape)



        plt.figure(figsize=(16, 6))

        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')

        X_test_2D = X.cpu().numpy()
        scatter = plt.scatter(X_test_2D[:, 0], X_test_2D[:, 1],
                              c=Y.cpu(), cmap='coolwarm', edgecolors='k')

        if named=='test':
            for idx, (x, y) in enumerate(X):
                plt.text(x, (y+0.35), str(idx), fontsize=8, ha='center', va='center')

            names = Model.test_dataset['Therapeutic_antibody']
            text_x = x_max + 0.5
            text_y_start = y_max
            line_height = (y_max - y_min) / max(30, len(names))  # space between lines

            for i, name in enumerate(names):
                plt.text(text_x, text_y_start - i * line_height,
                         f"[{i}] {name}", fontsize=9, va='top', color='black')

        plt.title("Decision Boundary in PCA 2D space")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.savefig('plot/contour0.4Ver2_' + named + '.png')
        plt.close()


    Contour(X_test_tensor,y_test, named = 'test')
    Contour(xb,yb, named ='TrainTest')

if False:
    from Bio import pairwise2
    from Bio.pairwise2 import format_alignment


    seq1 = 'EVQLVESGGGLVQPGGSLRLSCAASGYEFSRSWMNWVRQAPGKGLEWVGRIYPGDGDTNYSGKFKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARDGSSWDWYFDVWGQGTLVTVSS'
    seq2 = 'EVQLVESGGGLVQPGGSLRLSCAASGYEFSRSWMNWVRQAPGKGLEWVGRIYPGDGDTQYSGKFKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARDGSSWDWYFDVWGQGTLVTVSS'

    # Global alignment
    alignments = pairwise2.align.globalxx(seq1, seq2)

    # Similarity score (number of matches)
    print("Score:", alignments[0].score)
    if len(seq1)==alignments[0].score:
        print("Perfect match")
    else:
        print("Not a perfect match")

# Polar and Bar plot
if True:
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    All_data = np.vstack([X_train, X_test])
    All_labels = np.concatenate([y_train, y_test])
    colors = ['red' if label == 0 else 'blue' for label in All_labels]
    Z = [np.sqrt(pca[0]**2+pca[1]**2) for pca in All_data]

    # Create a circular plot
    theta = np.linspace(0.3, 2 * np.pi, len(Z))
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': 'polar'})
    ax.scatter(theta, Z, c=colors, s=50)
    r_min, r_max = min(Z), max(Z)
    ax.set_yticks(np.linspace(np.round(r_min), np.round(r_max), 4))
    ax.plot(theta, Z, linestyle='-', linewidth=0.5, color='gray')
    ax.tick_params(labelsize=14)
    legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Class 0', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='blue', markersize=10)]
    ax.legend(handles=legend_elements, loc='upper center', fontsize=12, frameon=True, edgecolor='black',framealpha=1,handletextpad=0.5,)
    ax.set_xticklabels([])
    ax.set_rlabel_position(0)
    ax.set_rlabel_position(0)
    ax.set_title("", va='bottom')
    plt.savefig('plot/polar.ver2.png')

    # Create a bar plot of Z
    fig, ax = plt.subplots()
    plt.bar(range(len(Z)), Z, color=colors)
    blue_patch = mpatches.Patch(color='blue', label='Class 0')
    red_patch = mpatches.Patch(color='red', label='Class 1')
    plt.legend(handles=[blue_patch, red_patch])
    ax.set_xlabel("Antibodies index")
    ax.set_ylabel("Distance")
    plt.savefig('plot/bar.ver2.png')

    # Two bars only
    X =  [pca[0]  for pca in All_data]
    Y =  [pca[1]  for pca in All_data]

if True:
    x_min, x_max = All_data_redu_unsupervised[:, 0].min() - 2, All_data_redu_unsupervised[:, 0].max() + 2
    y_min, y_max = All_data_redu_unsupervised[:, 1].min() - 2, All_data_redu_unsupervised[:, 1].max() + 2

    # Create grid
    xx, yy = np.meshgrid(np.linspace(x_min.item(), x_max.item(), 100),
                         np.linspace(y_min.item(), y_max.item(), 100))

    # Flatten and stack for prediction
    grid_2D = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
    All_labels = np.ones(grid_2D.shape[0])

    HierC.plot_class_oriented_dendrogram(
                            X=grid_2D,
                            y=All_labels,
                            method='ward', # ward, single, complete, average, median, weighted, centroid
                            metric='euclidean', # euclidean,  correlation, hamming, cosine, cityblock, chebyshev, jaccard
                            target_class=1,
                            push='right',
                            figsize=(18, 6),
                            leaf_font_size=7,
                            show_strip=True,
                            File_name= File_name + 'test'
    )

    mask = [tmp=='tab:blue' for tmp in HierC.dendro['color_list']]
    HierC.dendro['leaves']

    # ---- helpers over linkage ----
    def _children_map(Z):
        n = Z.shape[0] + 1
        return {n+i: (int(Z[i,0]), int(Z[i,1])) for i in range(n-1)}, n

    def _leaves_under(node, children, n):
        out, stack = [], [node]
        while stack:
            k = stack.pop()
            if k < n: out.append(k)
            else:     stack.extend(children[k])
        return out

    # ---- case 1: exactly 2 clusters (root split: left vs right) ----
    def clusters_root_split(Z):
        """Return an array c (length n) with 0 for left subtree, 1 for right subtree."""
        children, n = _children_map(Z)
        left_root, right_root = int(Z[-1,0]), int(Z[-1,1])
        left  = set(_leaves_under(left_root, children, n))
        right = set(_leaves_under(right_root, children, n))
        c = np.zeros(n, dtype=int)
        for i in right: c[i] = 1
        return c

    cluster_2 = clusters_root_split(HierC.Z).reshape(xx.shape)
    order = HierC.dendro['leaves']

#    cluster_2 = y_Pred.reshape(xx.shape)

    X_test_2D = All_data_redu.copy()
    Y = np.concatenate([y_train, y_test])



    plt.figure(figsize=(16, 6))
    plt.contourf(xx, yy, cluster_2, alpha=0.4, cmap='coolwarm')
    scatter = plt.scatter(X_test_2D[:, 0], X_test_2D[:, 1], c=Y, cmap='coolwarm', edgecolors='k')
    plt.title("Decision Boundary in PCA 2D space")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.savefig('plot/contour_fcluster' +  '.png')
    plt.close()


if True:
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a sample array of data (e.g., a normal distribution)
    data = ada_values

    # Plot the histogram
    # The 'bins' argument controls how many bars to use
    plt.figure(figsize=(16, 6))
    plt.hist(data, bins=100, edgecolor='black', alpha=0.7)

    # Add title and labels for clarity
    plt.title("Histogram of Sample Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Display the plot
    plt.savefig('plot/histogram.png')

if True:
    from sklearn.metrics import homogeneity_score, silhouette_score
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import linkage, fcluster

    # X: (n,d) features, y_true: ground truth labels (optional)
    X_std = StandardScaler().fit_transform(All_data_redu)

    # Example: cut a hierarchical tree at K clusters
    Z = linkage(X_std, method='ward')
    K = 4
    labels_pred = fcluster(Z, t=2, criterion='maxclust')

    # External (needs y_true)
    h = homogeneity_score(All_labels, labels_pred)  # 0..1

    # Internal (no labels needed)
    sil = silhouette_score(All_data_redu, labels_pred, metric='euclidean')  # -1..1
    print(f"silhouette_score is {sil}")
    print(f"Homogeneity: {h:.3f}, Silhouette: {sil:.3f}")
