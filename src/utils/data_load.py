import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = os.path.join("..", "data", "processed", "data_with_rolling_features_1.csv")

EXCLUDE_COLS = {"user", "date_only", "is_malicious", "user_id"}
TARGET_COL = "is_malicious"
TEST_SIZE = 0.25
RANDOM_STATE = 42
CUSTOM_THRESHOLD = 0.95
PLOT_OUTPUT = "pr_f1_xgb.png"

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset from the specified path, prepare features and target variable.
    parameter: path: The file path to the CSV dataset.
    returns: A tuple containing the features DataFrame and target Series.
    """
    df = pd.read_csv(DATA_PATH)
    df["date_only"] = pd.to_datetime(df["date_only"])
 
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]
    return X, y


def split_and_scale(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Split the dataset into training and testing sets, and scale the features.
    parameters:
        X: The features DataFrame.
        y: The target Series.
    returns: A tuple containing the scaled training features, scaled testing features, training target, and testing target.
    """
  
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
 