"""
src/cloud/train_xgboost.py

Run:
    python src/cloud/train_xgboost.py

Outputs (saved to models/):
    xgb_model.pkl       — trained XGBoost classifier
    feature_names.json  — ordered feature list (used by fog server)
"""

import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[2]
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# Load processed splits
def load_splits():
    X_train = pd.read_parquet(PROC_DIR / "X_train.parquet")
    X_val   = pd.read_parquet(PROC_DIR / "X_val.parquet")
    X_test  = pd.read_parquet(PROC_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROC_DIR / "y_train.parquet").squeeze()
    y_val   = pd.read_parquet(PROC_DIR / "y_val.parquet").squeeze()
    y_test  = pd.read_parquet(PROC_DIR / "y_test.parquet").squeeze()
    log.info(f"Loaded splits — train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# Evaluation helper
def evaluate(model, X, y, split_name: str) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auroc = roc_auc_score(y, y_prob)
    auprc = average_precision_score(y, y_prob)
    report = classification_report(y, y_pred, output_dict=True)

    metrics = {
        f"{split_name}_auroc":     round(auroc, 4),
        f"{split_name}_auprc":     round(auprc, 4),
        f"{split_name}_f1_pos":    round(report["1"]["f1-score"], 4),
        f"{split_name}_precision": round(report["1"]["precision"], 4),
        f"{split_name}_recall":    round(report["1"]["recall"], 4),
    }

    log.info(f"\n{'='*50}\n{split_name.upper()} results:")
    log.info(f"  AUROC: {auroc:.4f}  |  AUPRC: {auprc:.4f}")
    log.info(f"  F1 (mortality class): {report['1']['f1-score']:.4f}")
    log.info(f"\n{classification_report(y, y_pred)}")
    log.info(f"Confusion matrix:\n{confusion_matrix(y, y_pred)}")
    return metrics


# Train
def train():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    log.info(f"Class imbalance ratio (neg/pos): {scale_pos_weight:.2f} → used as scale_pos_weight")

    params = {
        "n_estimators":     500,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma":            0.1,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric":      "aucpr",
        "early_stopping_rounds": 30,
        "random_state":     42,
        "n_jobs":           -1,
    }

    mlflow.set_experiment("icu-deterioration")

    with mlflow.start_run(run_name="xgboost-baseline"):
        mlflow.log_params({k: v for k, v in params.items() if k != "eval_metric"})
        mlflow.log_param("features", X_train.shape[1])
        mlflow.log_param("train_samples", len(X_train))

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        # Evaluate on all splits
        val_metrics  = evaluate(model, X_val,  y_val,  "val")
        test_metrics = evaluate(model, X_test, y_test, "test")

        mlflow.log_metrics({**val_metrics, **test_metrics})
        mlflow.xgboost.log_model(model, artifact_path="xgb_model")

        # Feature importance
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        log.info(f"\nTop 15 features:\n{importance.head(15).round(4).to_string()}")

        # Save locally too
        model.save_model(MODEL_DIR / "xgb_model.json") 
        joblib.dump(model, MODEL_DIR / "xgb_model.pkl")
        with open(MODEL_DIR / "feature_names.json", "w") as f:
            json.dump(X_train.columns.tolist(), f, indent=2)

        log.info(f"\nModel saved → {MODEL_DIR / 'xgb_model.pkl'}")
        log.info(f"Best iteration: {model.best_iteration}")
        log.info(f"Val AUROC: {val_metrics['val_auroc']}  |  Test AUROC: {test_metrics['test_auroc']}")

    return model, importance


if __name__ == "__main__":
    train()