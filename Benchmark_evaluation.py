import numpy as np
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, adjusted_mutual_info_score
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import cophenet
import numpy.random as npr

# ---------- Dunn index (Euclidean by default) ----------
def dunn_index(X, labels, metric='euclidean'):
    X = np.asarray(X); labels = np.asarray(labels)
    D = squareform(pdist(X, metric=metric))
    clusters = [np.where(labels == k)[0] for k in np.unique(labels)]
    # intra = max cluster diameter
    intra = max(D[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].max() if len(idx) > 1 else 0.0
                for idx in clusters)
    # inter = min distance between points in different clusters
    inter = np.inf
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            Dij = D[np.ix_(clusters[i], clusters[j])]
            if Dij.size: inter = min(inter, Dij.min())
    return inter / intra if intra > 0 else np.inf


# ---------- Gap statistic (Euclidean) ----------
def gap_statistic(X, labels, B=20, random_state=0):
    """
    Computes Tibshirani gap at the provided clustering (fixed K = #unique labels).
    Wk = within-cluster dispersion (sum of squared distances to cluster centroids).
    """
    rng = npr.default_rng(random_state)
    X = np.asarray(X); labels = np.asarray(labels)
    K = len(np.unique(labels))

    def within_ss(X, labels):
        ss = 0.0
        for k in np.unique(labels):
            idx = (labels == k)
            if idx.sum() <= 1:
                continue
            Ck = X[idx]
            mu = Ck.mean(axis=0)
            ss += ((Ck - mu)**2).sum()
        return ss

    # observed
    Wk = within_ss(X, labels)
    logWk = np.log(Wk + 1e-12)

    # reference: uniform in bounding box
    mins, maxs = X.min(axis=0), X.max(axis=0)
    ref_logW = []
    for _ in range(B):
        Xref = rng.uniform(mins, maxs, size=X.shape)
        # assign ref points to nearest observed centroids for fair Wk_ref
        mus = np.vstack([X[labels == k].mean(axis=0) for k in np.unique(labels)])
        lab_ref = cdist(Xref, mus).argmin(axis=1)
        ref_logW.append(np.log(within_ss(Xref, lab_ref) + 1e-12))
    gap = np.mean(ref_logW) - logWk
    se_gap = np.std(ref_logW, ddof=1) * np.sqrt(1 + 1/B)
    return {"gap": float(gap), "gap_se": float(se_gap), "K": int(K)}


# ---------- Consensus stability (PAC) ----------
def consensus_pac_stability(X, clusterer, n_boot=20, sample_frac=0.8, random_state=0,
                            pac_bounds=(0.1, 0.9)):
    """
    clusterer: callable (X_subset, rng) -> labels_subset (0..K-1) for the subset order provided.
    Builds a co-association matrix over bootstraps and returns PAC = proportion of ambiguous pairs.
    Lower PAC -> better stability. Also returns the full consensus matrix.
    """
    rng = npr.default_rng(random_state)
    n = len(X)
    C = np.zeros((n, n), dtype=float)  # co-association counts
    N = np.zeros((n, n), dtype=float)  # #times a pair was co-sampled

    idx_all = np.arange(n)
    for b in range(n_boot):
        m = max(2, int(sample_frac * n))
        S = rng.choice(idx_all, size=m, replace=False)
        labels_S = clusterer(X[S], rng)
        # update pair counts cluster by cluster
        for k in np.unique(labels_S):
            ids = S[labels_S == k]
            C[np.ix_(ids, ids)] += 1.0
        N[np.ix_(S, S)] += 1.0

    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(N > 0, C / N, 0.0)

    l, u = pac_bounds
    mask = (N > 0) & (P > l) & (P < u)
    pac = P[mask].size / max(1, (N > 0).sum())
    return {"pac": float(pac), "consensus": P}

# ---------- Cophenetic correlation (for hierarchical) ----------
def cophenetic_corr(Z, X, metric='euclidean'):
    """
    Z: linkage matrix; X: original data used to build Z
    """
    from scipy.spatial.distance import pdist
    c, _ = cophenet(Z)          # cophenetic distances from the tree
    d = pdist(X, metric=metric) # original pairwise distances
    # Pearson correlation between c and d
    num = np.corrcoef(c, d)[0,1]
    return float(num)

# ---------- Simple wrapper to compute a set of metrics ----------
def evaluate_clustering(X, labels, Z=None, y_true=None, random_state=0):
    out = {}
    X = np.asarray(X); labels = np.asarray(labels)

    # internal
    out["Davies_Bouldin"] = float(davies_bouldin_score(X, labels))
    out["Calinski_Harabasz"] = float(calinski_harabasz_score(X, labels))
    out["Dunn"] = float(dunn_index(X, labels))
    out["Gap"] = gap_statistic(X, labels, B=20, random_state=random_state)

    # external (optional)
    if y_true is not None:
        y_true = np.asarray(y_true)
        out["ARI"] = float(adjusted_rand_score(y_true, labels))
        out["AMI"] = float(adjusted_mutual_info_score(y_true, labels))

    # dendrogram fidelity (optional)
    if Z is not None:
        out["Cophenetic_corr"] = cophenetic_corr(Z, X)

    return out



import numpy as np
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             v_measure_score, fowlkes_mallows_score,
                             roc_auc_score, average_precision_score,
                             brier_score_loss, log_loss,
                             accuracy_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef)

# ----- Variation of Information (lower is better) -----
def variation_of_information(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    ut, up = np.unique(y_true), np.unique(y_pred)
    # contingency (proportions)
    P = np.zeros((len(ut), len(up)), float)
    for i, t in enumerate(ut):
        for j, c in enumerate(up):
            P[i, j] = np.sum((y_true == t) & (y_pred == c)) / n
    pt = P.sum(axis=1); pp = P.sum(axis=0)
    # entropies + mutual information (nats)
    Ht = -np.sum(pt[pt > 0] * np.log(pt[pt > 0]))
    Hp = -np.sum(pp[pp > 0] * np.log(pp[pp > 0]))
    I  = 0.0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] > 0:
                I += P[i, j] * np.log(P[i, j] / (pt[i] * pp[j]))
    return Ht + Hp - 2 * I

# ----- Probabilities from cluster composition -----
def cluster_prob_scores(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)
    # p_hat per cluster
    p_hat = {}
    for c in np.unique(y_pred):
        m = (y_pred == c)
        p = y_true[m].mean() if m.any() else 0.0
        p_hat[c] = np.clip(p, eps, 1 - eps)
    y_score = np.array([p_hat[c] for c in y_pred])
    # metrics
    try:
        auc = roc_auc_score(y_true, y_score)
        ap  = average_precision_score(y_true, y_score)
        ll  = log_loss(y_true, y_score)
    except Exception:
        auc, ap, ll = np.nan, np.nan, np.nan
    brier = brier_score_loss(y_true, y_score)
    return {"ROC_AUC": float(auc), "PR_AUC": float(ap),
            "Brier": float(brier), "LogLoss": float(ll)}

# ----- Majority mapping → binary predictions -----
def majority_map_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)
    y_hat = np.empty_like(y_true)
    for c in np.unique(y_pred):
        m = (y_pred == c)
        # majority class in this cluster (tie -> class 0)
        vals, counts = np.unique(y_true[m], return_counts=True)
        y_hat[m] = vals[np.argmax(counts)]
    return {
        "Accuracy": float(accuracy_score(y_true, y_hat)),
        "BalancedAcc": float(balanced_accuracy_score(y_true, y_hat)),
        "F1": float(f1_score(y_true, y_hat)),
        "MCC": float(matthews_corrcoef(y_true, y_hat)),
    }

# ----- One function to report the key scores -----
def evaluate_binary_truth_vs_n_clusters(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)

    out = {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "AMI": float(adjusted_mutual_info_score(y_true, y_pred)),
        "V_measure": float(v_measure_score(y_true, y_pred)),
        "FMI": float(fowlkes_mallows_score(y_true, y_pred)),
        "VI": float(variation_of_information(y_true, y_pred)),   # lower is better
    }
    out.update(cluster_prob_scores(y_true, y_pred))
    out.update(majority_map_metrics(y_true, y_pred))
    return out



# Z_re is your (optionally) reordered linkage that pushes class-1 right
Z_re = HierC.Z
y = All_labels

fig, ax = plt.subplots(figsize=(20, 6))
dendro, groups_list = color_by_second_split(Z_re, y=All_labels,  ax=ax, leaf_font_size=7,
                                       cluster_colors=('tab:blue','tab:orange','tab:green','tab:red'),
                                       show_cluster_strip=True,
                                       keep_leaf_class_colors=True)

leaf_order = dendro['leaves']


cluster_order_class = y[leaf_order]

cluste1 = np.array([tmp=='tab:blue' for tmp in dendro['leaves_color_list']]).astype(int)*1
cluste2 = np.array([tmp=='tab:orange' for tmp in dendro['leaves_color_list']]).astype(int)*2
cluste3 = np.array([tmp=='tab:green' for tmp in dendro['leaves_color_list']]).astype(int)*3
cluste4 = np.array([tmp=='tab:red' for tmp in dendro['leaves_color_list']]).astype(int)*4
y_Pred = cluste1+cluste2+cluste3+cluste4-1

# y_true: 0/1 ground truth, y_pred: cluster ids (0..K-1 or arbitrary ints)
scores = evaluate_binary_truth_vs_n_clusters(cluster_order_class, y_Pred)
for k, v in scores.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
