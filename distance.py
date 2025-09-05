import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def average_cross_distance(
    X, y, metric='euclidean', standardize=True,
    sample_pairs=None, random_state=0
):
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
      - energy_like = 2*between - within_0 - within_1
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
    energy_like = 2*mean_between - (mean_within_0 + mean_within_1)

    # Scale-free ratio (larger is better)
    denom = (mean_within_0 + mean_within_1) / 2 if (mean_within_0 + mean_within_1) > 0 else np.nan
    ratio = mean_between / denom if denom and not np.isnan(denom) else np.inf

    # Silhouette (only well-defined for Euclidean here)
    sil = silhouette_score(X, y) if metric == 'euclidean' and len(np.unique(y)) == 2 else None

    return {
        "mean_between": mean_between,
        "mean_within_0": mean_within_0,
        "mean_within_1": mean_within_1,
        "energy_like": energy_like,
        "ratio": ratio,
        "silhouette": sil,
        "nA": len(A_sub), "nB": len(B_sub)
    }



def best_cut_by_distance_score(scores, X, metric='euclidean',
                               standardize=True, n_cuts=200,
                               objective='ratio', sample_pairs=None, random_state=0):
    """
    Sweep cutoffs on 'scores' to form y = (scores >= t), compute separation,
    and return the best cutoff by the chosen objective:
      - 'between'  -> maximize mean_between
      - 'energy'   -> maximize energy_like
      - 'ratio'    -> maximize between / mean(within)
      - 'sil'      -> maximize silhouette (euclidean only)
    """
    scores = np.asarray(scores).ravel()
    # candidate cutoffs between quantiles to avoid degenerate splits
    qs = np.linspace(0.05, 0.95, n_cuts)
    ts = np.quantile(scores, qs)

    best = {"score": -np.inf, "t": None, "result": None}
    for t in ts:
        y = (scores >= t).astype(int)
        if y.min() == y.max():  # all one class
            continue
        res = average_cross_distance(
            X, y, metric=metric, standardize=standardize,
            sample_pairs=sample_pairs, random_state=random_state
        )
        val = {
            'between':  res["mean_between"],
            'energy':   res["energy_like"],
            'ratio':    res["ratio"],
            'sil':      res["silhouette"] if res["silhouette"] is not None else -np.inf
        }[objective]
        if val is not None and val > best["score"]:
            best = {"score": val, "t": t, "result": res}
    return best



# X: (n, d) features; y: 0/1 groups (e.g., red=1, blue=0)
out = average_cross_distance(All_data_redu_unsupervised, All_labels, metric='euclidean', standardize=True)
print(out)
# {'mean_between': ..., 'mean_within_0': ..., 'mean_within_1': ...,
#  'energy_like': ..., 'ratio': ..., 'silhouette': ..., 'nA': ..., 'nB': ...}

# If you have a continuous score to define the split:
best = best_cut_by_distance_score(scores, X, metric='euclidean', objective='ratio')
print("Best cutoff:", best["t"], "score:", best["score"])
print("Details:", best["result"])
