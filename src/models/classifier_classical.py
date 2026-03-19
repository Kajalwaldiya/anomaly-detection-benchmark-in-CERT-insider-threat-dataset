import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score

from src.utils.data_load import load_data
from src.utils.data_load import split_and_scale, RANDOM_STATE

RESULTS_PATH = os.path.join("..", "results")

# Define param grid     
def build_xgb_param_grid(y: pd.Series) -> dict:
    benign_ratio = (y == 0).sum() / (y == 1).sum()
    return {
        "n_estimators": [100, 200],
        "learning_rate": [0.1, 0.05],
        "scale_pos_weight": [int(benign_ratio), int(benign_ratio * 1.5)],
    }
 
def train_xgb_with_grid_search(X_train_scaled, y_train, param_grid):
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Grid search
    grid = GridSearchCV(
        estimator=xgb, 
        param_grid=param_grid, 
        scoring='roc_auc', 
        cv=3, 
        verbose=2, 
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)

    # Best model
    best_model = grid.best_estimator_
    print("Best Parameters:", grid.best_params_)

    return grid.best_estimator_


def train_random_forest(
    X_train: np.ndarray, y_train: pd.Series
) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    return rf
 

def evaluate_model(X_test_scaled, y_test, best_model, model_name="xgb"):
    # Evaluate
    preds = best_model.predict(X_test_scaled)
    probs = best_model.predict_proba(X_test_scaled)[:, 1]
    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("AUC:", roc_auc_score(y_test, probs))



    # Predict probabilities using your tuned model
    probs = best_model.predict_proba(X_test_scaled)[:, 1]
    y_true = y_test.values

    # Thresholds for evaluation
    thresholds = np.linspace(0.0, 1.0, 200)

    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, probs)

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, probs)

    # F1 scores at each threshold
    f1_scores = [f1_score(y_true, probs >= t) for t in thresholds]

    # Plot Precision, Recall, and F1 vs Threshold
    plt.figure(figsize=(14, 5))

    # PR Curve
    plt.subplot(1, 2, 1)
    plt.plot(pr_thresholds, precision[:-1], label="Precision")
    plt.plot(pr_thresholds, recall[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision & Recall vs Threshold")
    plt.legend()
    plt.grid(True)

    # F1 Curve
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, f1_scores, label="F1 Score", color='green')
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Threshold")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f'pr_f1_{model_name}.png'))



    # Custom threshold
    threshold = 0.95
    custom_preds = (probs >= threshold).astype(int)

    # Evaluate
    print(f"Classification Report @ threshold=0.95 for {model_name}:")
    print(classification_report(y_test, custom_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, custom_preds))
    print("AUC:", roc_auc_score(y_test, probs))


# Load your data prepare features and target
X, y, _ = load_data()
X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(X, y)

param_grid = build_xgb_param_grid(y_train)

best_model = train_xgb_with_grid_search(X_train_scaled, y_train, param_grid)

evaluate_model(X_test_scaled, y_test, best_model)

rf_model = train_random_forest(X_train_scaled, y_train)
evaluate_model(X_test_scaled, y_test, rf_model, model_name="rf")
