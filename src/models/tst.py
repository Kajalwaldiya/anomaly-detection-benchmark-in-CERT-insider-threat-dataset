import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tsai.all import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score

from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from src.utils.data_load import load_data, EXCLUDE_COLS
from src.utils.process_data import data_preparation_ts
from sklearn.metrics import precision_recall_curve



RESULTS_PATH = os.path.join("..", "results")

dls, x_valid, y_valid = data_preparation_ts()

# 4. Create Learner with MLSTM-FCN
learn = ts_learner(dls, arch='TST', metrics=[accuracy, F1Score(average='macro')])

# 5. Train the model
learn.fit_one_cycle(20, lr_max=1e-3, wd=1e-2)

# 6. Evaluate
learn.show_results()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
# interp.plot_top_losses(10)

# 1. Classification Interpretation
interp = ClassificationInterpretation.from_learner(learn)

# 2. Save Confusion Matrix
fig_cm, ax_cm = plt.subplots()
interp.plot_confusion_matrix(ax_cm)
fig_cm.savefig(os.path.join(RESULTS_PATH, "confusion_matrix_tst.png"), bbox_inches="tight")

# 3. Save ROC Curve
preds, targs, decoded = learn.get_preds(dl=dls.valid, with_decoded=True)

y_true = targs.numpy()
y_proba = preds[:, 1].numpy()
# y_true = interp.y_true
# y_proba = interp.preds[:, 1]  # Probabilities for class 1
fpr, tpr, thresholds = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend()
ax_roc.grid(True)
fig_roc.savefig(os.path.join(RESULTS_PATH, "roc_curve_tst.png"), bbox_inches="tight")

# 4. Save Loss Curve
fig_loss, ax_loss = plt.subplots()
learn.recorder.plot_loss(ax=ax_loss)
fig_loss.savefig(os.path.join(RESULTS_PATH, "loss_curve_tst.png"), bbox_inches="tight")


# Get predictions from validation set
preds, targs, decoded = learn.get_preds(dl=dls.valid, with_decoded=True)

# Classification report (true labels vs predicted class)
y_true = targs.numpy()
y_pred = decoded.numpy()

target_names = [str(c) for c in np.unique(y_true)]


print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))


# Probabilities for positive class
y_proba = preds[:, 1].numpy()

# Precision-recall curve
precision, recall, _ = precision_recall_curve(y_true, y_proba)
ap_score = average_precision_score(y_true, y_proba)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f'AP = {ap_score:.2f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_PATH, "pr_curve_tst.png"), bbox_inches="tight")