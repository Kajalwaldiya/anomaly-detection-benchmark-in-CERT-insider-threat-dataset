from tsai.all import *
import numpy as np
import os
import matplotlib.pyplot as plt

from src.utils.data_load import load_data, EXCLUDE_COLS
from src.utils.data_load import split_and_scale
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay, accuracy_score
)

RESULTS_PATH = os.path.join("..", "results")

X, y, df = load_data()
# Preprocessing
df["date_only"] = pd.to_datetime(df["date_only"])
df = df.sort_values(["user", "date_only"])

# Parameters
lookback = 7

# Define feature columns (exclude identifiers and labels)
exclude_cols = ['user', 'date_only', 'is_malicious']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Normalize features per user
scalers = {}
for user in df['user'].unique():
    mask = df['user'] == user
    scalers[user] = MinMaxScaler()
    df.loc[mask, feature_cols] = scalers[user].fit_transform(df.loc[mask, feature_cols])

# Create sequences
X, y = [], []
user_sequences = []

for user_id, group in df.groupby('user'):
    group = group.sort_values('date_only')
    if len(group) < lookback:
        continue
    for i in range(len(group) - lookback + 1):
        seq = group.iloc[i:i + lookback]
        X.append(seq[feature_cols].values.T)  # Shape: (features, sequence_length)
        y.append(seq['is_malicious'].iloc[-1])
        user_sequences.append(user_id)

X = np.stack(X)
y = np.array(y)
user_sequences = np.array(user_sequences)

# 1. Split Time-Aware: 80% train, 20% valid
split_index = int(len(X) * 0.8)
X_train, X_valid = X[:split_index], X[split_index:]
y_train, y_valid = y[:split_index], y[split_index:]

X_combined = np.concatenate([X_train, X_valid])
y_combined = np.concatenate([y_train, y_valid])


# 2. Create Datasets with TSAI
splits = (list(range(len(X_train))), list(range(len(X_train), len(X_combined))))

# 3. Create datasets
tfms = [None, [Categorize()]]
datasets = TSDatasets(X_combined, y_combined, tfms=tfms, splits=splits)

# 3. Create DataLoaders
dls = TSDataLoaders.from_dsets(datasets.train, datasets.valid, bs=64)

# 4. Create Learner with MLSTM-FCN
learn = ts_learner(dls, arch='MLSTM_FCN', metrics=[accuracy, F1Score(average='macro')])

# 5. Train the model
learn.fit_one_cycle(20, lr_max=1e-3, wd=1e-2)

# 6. Evaluate
# learn.show_results()
interp = ClassificationInterpretation.from_learner(learn)
print(interp.plot_confusion_matrix())
# interp.plot_top_losses(10)

# 1. Classification Interpretation
print("Classification Report:")
print(classification_report(y_valid, interp.preds.argmax(dim=1)))
# 2. Save Confusion Matrix
fig_cm, ax_cm = plt.subplots()
interp.plot_confusion_matrix(ax_cm)
fig_cm.savefig(os.path.join(RESULTS_PATH, "confusion_matrix_mlstm_fcn.png"), bbox_inches="tight")

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
fig_roc.savefig(os.path.join(RESULTS_PATH, "roc_curve_mlstm_fcn.png"), bbox_inches="tight")

# 4. Save Loss Curve
fig_loss, ax_loss = plt.subplots()
learn.recorder.plot_loss(ax=ax_loss)
fig_loss.savefig(os.path.join(RESULTS_PATH, "loss_curve_mlstm_fcn.png"), bbox_inches="tight")
