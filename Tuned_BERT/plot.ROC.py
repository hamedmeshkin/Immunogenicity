import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

All_y_true = []
All_y_scores = []
for inx in range(len(Model.predictors)):
    # Get predicted probabilities for the positive class (label=1)
    y_true = data_test['Label'].values
    y_scores = Model.predictors[inx].predict_proba(data_test)[1].values  # column for class 1

    All_y_true.append(y_true)
    All_y_scores.append(y_scores)
    # Compute ROC curve
All_y_scores = np.stack(All_y_scores).mean(0)
All_y_true = y_true
fpr, tpr, thresholds = roc_curve(All_y_true, All_y_scores)
roc_auc = roc_auc_score(All_y_true, All_y_scores)

# Find threshold=0.5 index
thresh_idx = np.argmin(np.abs(thresholds - 0.5))
threshold_point = (fpr[thresh_idx], tpr[thresh_idx])

# Plot
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, linestyle='--', color='red', linewidth=3, label=f'(AUROC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)

# Annotate threshold 0.5
plt.scatter(*threshold_point, color='darkred', s=80)
plt.text(threshold_point[0]+0.02, threshold_point[1]-0.05, 'Threshold:0.5', fontsize=10)

# Final plot settings
plt.title('ROC curves')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('plot/AUC')
