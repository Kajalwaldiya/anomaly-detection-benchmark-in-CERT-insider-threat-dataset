from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from src.utils.data_load import load_data, EXCLUDE_COLS
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
from tensorflow.keras.layers import Dropout


# change ratio here
RATIO = 0.005

print("lstm-ae sequencing")
# Load data
X, y, df = load_data()
df['date_only'] = pd.to_datetime(df['date_only'])
feature_cols = df.drop(columns=EXCLUDE_COLS).columns.tolist()

# Efficient User-wise Min-Max Scaling
def userwise_minmax_scale(df, feature_cols):
    scaled_data = []
    for user, group in df.groupby("user"):
        scaler = MinMaxScaler()
        group_scaled = group.copy()
        if len(group) > 1:
            group_scaled[feature_cols] = scaler.fit_transform(group[feature_cols])
        else:
            group_scaled[feature_cols] = 0  # Edge case: only one row
        scaled_data.append(group_scaled)
    return pd.concat(scaled_data)

df_scaled = userwise_minmax_scale(df, feature_cols)

# Sequence Construction
def create_sequences(df, feature_cols, window_size=10):
    sequences, labels = [], []
    for _, group in df.groupby("user"):
        values = group[feature_cols].values
        labels_ = group['is_malicious'].values
        for i in range(len(values) - window_size + 1):
            sequences.append(values[i:i+window_size])
            labels.append(labels_[i+window_size-1])
    return np.array(sequences), np.array(labels)

X, y = create_sequences(df_scaled, feature_cols, window_size=3)


# Split data
normal_idx = np.where(y == 0)[0]
anomaly_idx = np.where(y == 1)[0]

train_size = int(0.7 * len(normal_idx))
val_size = int(0.1 * len(normal_idx))



# Keep training as is
X_train = X[normal_idx[:train_size]]

# n_anom_val = int(0.1 * len(anomaly_idx))  # 10% of anomalies

# Keep training as is 20%

X_train = X[normal_idx[:train_size]]
n_anom_val = int(0.1 * len(anomaly_idx))  # 10% of anomalies

# Validation (10% normal + 10% anomaly)
X_val_norm = X[normal_idx[train_size:train_size+val_size]]
X_val_anom = X[anomaly_idx[:n_anom_val]]
X_val = np.concatenate([X_val_norm, X_val_anom])
y_val = np.array([0]*len(X_val_norm) + [1]*len(X_val_anom))

# Test: 20% anomalies, 80% normals
X_test_anom = X[anomaly_idx[n_anom_val:]]
y_test_anom = y[anomaly_idx[n_anom_val:]]


### desired ratio
def create_test_set_with_ratio(X, y, normal_idx, anomaly_idx, start_anom_idx, ratio=0.1):
    """
    Create a test set with a specific anomaly ratio (e.g. 0.1 for 10%)
    """
    # Remaining anomalies for test
    X_anom = X[anomaly_idx[start_anom_idx:]]
    y_anom = y[anomaly_idx[start_anom_idx:]]
    n_anom = len(X_anom)

    # How many anomalies to include
    target_anom_ratio = ratio
    n_test_anom = min(n_anom, 5000)  # Cap to prevent excessive size

    # Required # normal samples to achieve desired ratio
    n_test_norm = int((n_test_anom / target_anom_ratio) - n_test_anom)

    # Sample normal data for test
    norm_start = train_size + val_size
    X_norm = X[normal_idx[norm_start:norm_start + n_test_norm]]
    y_norm = y[normal_idx[norm_start:norm_start + n_test_norm]]

    # Clip anomaly samples if needed
    X_anom_final = X_anom[:n_test_anom]
    y_anom_final = y_anom[:n_test_anom]

    # Combine
    X_test = np.concatenate([X_norm, X_anom_final])
    y_test = np.concatenate([y_norm, y_anom_final])
    
    return X_test, y_test

X_test, y_test = create_test_set_with_ratio(X, y, normal_idx, anomaly_idx, start_anom_idx=n_anom_val, ratio=RATIO)


# Model
timesteps = X_train.shape[1]
n_features = X_train.shape[2]

input_layer = Input(shape=(timesteps, n_features))

# Encoder
x = LSTM(32, activation='relu', return_sequences=True)(input_layer)
x = LSTM(16, activation='relu', return_sequences=False,
         activity_regularizer=regularizers.l2(1e-5))(x)

# Decoder
x = RepeatVector(timesteps)(x)
x = LSTM(16, activation='relu', return_sequences=True)(x)
x = LSTM(32, activation='relu', return_sequences=True)(x)
output = TimeDistributed(Dense(n_features))(x)

autoencoder = Model(inputs=input_layer, outputs=output)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')


print("MODEL: ")
print(autoencoder.summary())

# Train
history = autoencoder.fit(X_train, X_train,
                          epochs=95,
                          batch_size=256,
                          validation_data=(X_val, X_val),
                          shuffle=True,
                          verbose=1)

# Test
X_pred = autoencoder.predict(X_test)
mse = np.mean(np.square(X_pred - X_test), axis=(1, 2))

# Threshold
# threshold = 1.5 * np.mean(mse)
# y_pred = (mse > threshold).astype(int)
X_test_pred = autoencoder.predict(X_test)
recon_errors = np.mean(np.square(X_test_pred - X_test), axis=(1, 2))

# Step 2: Plot histograms for reconstruction errors (normal vs anomaly)
normal_errors = recon_errors[y_test == 0]
anomaly_errors = recon_errors[y_test == 1]

plt.figure(figsize=(10, 5))
plt.hist(normal_errors, bins=100, alpha=0.6, label='Normal', color='blue')
plt.hist(anomaly_errors, bins=100, alpha=0.6, label='Anomaly', color='red')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(str(RATIO*100)+"_seq_lstm_ae_recon_err.png")


def compute_tpr_tnr(y_true, errors, thresholds):
    tprs, tnrs = [], []
    for thresh in thresholds:
        y_pred = (errors > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn + 1e-8)
        tnr = tn / (tn + fp + 1e-8)
        tprs.append(tpr)
        tnrs.append(tnr)
    return tprs, tnrs

# # Sweep thresholds across observed range
thresholds = np.linspace(min(recon_errors), max(recon_errors), 1000)
tprs, tnrs = compute_tpr_tnr(y_test, recon_errors, thresholds)

# Find threshold where TPR and TNR intersect (minimum absolute difference)
best_idx = np.argmin(np.abs(np.array(tprs) - np.array(tnrs)))
best_threshold = thresholds[best_idx]
print("best threshold: ", best_threshold)
y_pred = (recon_errors > best_threshold).astype(int)


tprs, tnrs = compute_tpr_tnr(y_test, recon_errors, thresholds)
# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, mse))

# ROC and PR Curves
fpr, tpr, _ = roc_curve(y_test, mse)
prec, rec, _ = precision_recall_curve(y_test, mse)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, mse):.2f}")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(rec, prec)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.savefig(str(RATIO*100)+"_roc_and_pr_lstm_ae.png")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
fpr = 1 - tnr


# Print results
print("Best threshold where TPR ≈ TNR: ", best_threshold)
print(f"TPR (Recall): {tpr * 100:.2f}%")
print(f"TNR (Specificity): {tnr * 100:.2f}%")
print(f"FPR: {fpr * 100:.2f}%")

# Plot TPR vs TNR intersection
plt.figure(figsize=(8, 5))
plt.plot(thresholds, tprs, label='TPR (Recall)')
plt.plot(thresholds, tnrs, label='TNR (Specificity)')
plt.axvline(best_threshold, color='black', linestyle='--', label='Best Threshold')
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.title('TPR and TNR vs Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(str(RATIO*100)+"_tpr_vs_tnr_vs_th_lstm_ae.png")

losses = history.history
print("--------------------")
print(history)
print("--------------------")
print(losses)

val_loss = losses['val_loss']
loss = losses['loss']
plt.figure(figsize=(8, 5))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('loss')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.tight_layout()
plt.savefig(str(RATIO*100)+"_losses.png")

plt.figure(figsize=(8, 5))
plt.plot(loss, label='Training Loss')
plt.legend(loc='upper right')
plt.ylabel('loss')
plt.title('Training loss')
plt.xlabel('epoch')
plt.tight_layout()
plt.savefig(str(RATIO*100)+"10_tr_losses.png")
