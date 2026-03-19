from data_load import load_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tsai.all import *


def data_preparation_ts():
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

    dls = TSDataLoaders.from_dsets(datasets.train, datasets.valid, bs=64)

    return dls, X_valid, y_valid