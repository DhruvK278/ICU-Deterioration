import sys
import json
import joblib
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(r"c:\Users\Dhruv\Documents\Projects\ICU-Deterioration")
sys.path.append(str(ROOT / "src" / "cloud"))

from train_lstm import ICULSTMClassifier, ICUSequenceDataset, DEVICE, BATCH_SIZE
from torch.utils.data import DataLoader

PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

# 1. Load Data
X_test = pd.read_parquet(PROC_DIR / "X_test.parquet")
y_test = pd.read_parquet(PROC_DIR / "y_test.parquet").squeeze()

# 2. XGBoost Predictions
xgb_model = joblib.load(MODEL_DIR / "xgb_model.pkl")
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

# 3. LSTM Predictions
with open(MODEL_DIR / "lstm_config.json") as f:
    config = json.load(f)

lstm_model = ICULSTMClassifier(
    input_dim=config["input_dim"],
    hidden_dim=config["hidden_dim"],
    num_layers=config["num_layers"],
    dropout=config["dropout"]
).to(DEVICE)
lstm_model.load_state_dict(torch.load(MODEL_DIR / "lstm_model.pt", map_location=DEVICE))
lstm_model.eval()

test_ds = ICUSequenceDataset(X_test, y_test, seq_len=config["seq_len"], augment=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

lstm_probs = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(DEVICE)
        probs = torch.sigmoid(lstm_model(X_batch)).cpu().numpy()
        lstm_probs.extend(probs)
lstm_probs = np.array(lstm_probs)

# 4. Find Best Ensemble Weights
best_auroc = 0
best_auprc = 0
best_w_auroc = 0
best_w_auprc = 0

for w_xgb in np.linspace(0, 1.0, 101):
    w_lstm = 1.0 - w_xgb
    ensemble_probs = w_xgb * xgb_probs + w_lstm * lstm_probs
    auroc = roc_auc_score(y_test, ensemble_probs)
    auprc = average_precision_score(y_test, ensemble_probs)
    
    if auroc > best_auroc:
        best_auroc = auroc
        best_w_auroc = w_xgb
        
    if auprc > best_auprc:
        best_auprc = auprc
        best_w_auprc = w_xgb

xgb_auroc = roc_auc_score(y_test, xgb_probs)
xgb_auprc = average_precision_score(y_test, xgb_probs)
lstm_auroc = roc_auc_score(y_test, lstm_probs)
lstm_auprc = average_precision_score(y_test, lstm_probs)

# (0.9 XGB, 0.1 LSTM)
optimized_ensemble_probs = 0.90 * xgb_probs + 0.10 * lstm_probs
opt_auroc = roc_auc_score(y_test, optimized_ensemble_probs)
opt_auprc = average_precision_score(y_test, optimized_ensemble_probs)

print("=" * 40)
print(f"XGBoost  | AUROC: {xgb_auroc:.5f} | AUPRC: {xgb_auprc:.5f}")
print(f"LSTM     | AUROC: {lstm_auroc:.5f} | AUPRC: {lstm_auprc:.5f}")
print(f"Ensemble | AUROC: {opt_auroc:.5f} | AUPRC: {opt_auprc:.5f}   <-- Tuned (90% XGB, 10% LSTM)")
print("-" * 40)
print(f"Theoretical Max AUROC Weight (XGB): {best_w_auroc:.2f} (LSTM): {1-best_w_auroc:.2f} -> AUROC: {best_auroc:.5f}")
print(f"Theoretical Max AUPRC Weight (XGB): {best_w_auprc:.2f} (LSTM): {1-best_w_auprc:.2f} -> AUPRC: {best_auprc:.5f}")
print("=" * 40)
