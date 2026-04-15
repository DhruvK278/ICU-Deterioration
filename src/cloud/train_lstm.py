"""
src/cloud/train_lstm.py


Since mimic3d.csv contains one row per admission (not time-series),
we simulate a rolling window by constructing pseudo-sequences using
LOSgroupNum as the time axis and stratifying by acuity bands.
This mirrors what the fog server will do in production: accumulate
5-minute aggregated feature windows per patient and score them.

Run:
    python src/cloud/train_lstm.py

Outputs:
    models/lstm_model.pt        — trained PyTorch LSTM
    models/lstm_config.json     — architecture config for fog server
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[2]
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN    = 4      # number of time steps per sequence window
BATCH_SIZE = 256
EPOCHS     = 30
LR         = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT    = 0.3


# Dataset 
class ICUSequenceDataset(Dataset):
    """
    Wraps flat admission features into (SEQ_LEN, features) windows.

    In production the fog server accumulates real 5-min windows;
    here we simulate by repeating the admission vector with small
    Gaussian jitter across SEQ_LEN steps — the model still learns
    to score a sequence of feature vectors per patient.
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series, seq_len: int = SEQ_LEN, augment: bool = True):
        self.X       = torch.tensor(X.values, dtype=torch.float32)
        self.y       = torch.tensor(y.values, dtype=torch.float32)
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_base = self.X[idx]                            # (features,)
        if self.augment:
            # Add small noise to simulate temporal variation across steps
            noise  = torch.randn(self.seq_len, x_base.shape[0]) * 0.05
            x_seq  = x_base.unsqueeze(0).repeat(self.seq_len, 1) + noise
        else:
            x_seq  = x_base.unsqueeze(0).repeat(self.seq_len, 1)
        return x_seq, self.y[idx]                       # (seq_len, features), scalar


# Model
class ICULSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        last    = out[:, -1, :]          # take last time step
        return self.head(last).squeeze(1)


# Training loop
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        probs   = torch.sigmoid(model(X_batch)).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y_batch.numpy())
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    return auroc, auprc


# Main
def train():
    X_train = pd.read_parquet(PROC_DIR / "X_train.parquet")
    X_val   = pd.read_parquet(PROC_DIR / "X_val.parquet")
    X_test  = pd.read_parquet(PROC_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROC_DIR / "y_train.parquet").squeeze()
    y_val   = pd.read_parquet(PROC_DIR / "y_val.parquet").squeeze()
    y_test  = pd.read_parquet(PROC_DIR / "y_test.parquet").squeeze()

    input_dim = X_train.shape[1]
    log.info(f"Input dim: {input_dim}  |  Device: {DEVICE}")

    train_ds = ICUSequenceDataset(X_train, y_train, augment=True)
    val_ds   = ICUSequenceDataset(X_val,   y_val,   augment=False)
    test_ds  = ICUSequenceDataset(X_test,  y_test,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = ICULSTMClassifier(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    mlflow.set_experiment("icu-deterioration")

    with mlflow.start_run(run_name="lstm-fog-model"):
        mlflow.log_params({
            "hidden_dim":  HIDDEN_DIM,
            "num_layers":  NUM_LAYERS,
            "dropout":     DROPOUT,
            "seq_len":     SEQ_LEN,
            "epochs":      EPOCHS,
            "lr":          LR,
            "batch_size":  BATCH_SIZE,
        })

        best_val_auroc = 0.0
        patience_count = 0
        PATIENCE       = 7

        for epoch in range(1, EPOCHS + 1):
            train_loss          = train_epoch(model, train_loader, optimizer, criterion)
            val_auroc, val_auprc = evaluate(model, val_loader)
            scheduler.step(1 - val_auroc)

            mlflow.log_metrics({
                "train_loss": round(train_loss, 4),
                "val_auroc":  round(val_auroc, 4),
                "val_auprc":  round(val_auprc, 4),
            }, step=epoch)

            log.info(
                f"Epoch {epoch:02d}/{EPOCHS}  "
                f"loss: {train_loss:.4f}  "
                f"val_auroc: {val_auroc:.4f}  "
                f"val_auprc: {val_auprc:.4f}"
            )

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_count = 0
                torch.save(model.state_dict(), MODEL_DIR / "lstm_model.pt")
                log.info(f"  -> New best model saved (val_auroc={val_auroc:.4f})")
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    log.info(f"Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(torch.load(MODEL_DIR / "lstm_model.pt", map_location=DEVICE))
        test_auroc, test_auprc = evaluate(model, test_loader)
        log.info(f"\nTest AUROC: {test_auroc:.4f}  |  Test AUPRC: {test_auprc:.4f}")

        mlflow.log_metrics({
            "best_val_auroc": round(best_val_auroc, 4),
            "test_auroc":     round(test_auroc, 4),
            "test_auprc":     round(test_auprc, 4),
        })

        config = {
            "input_dim":  input_dim,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout":    DROPOUT,
            "seq_len":    SEQ_LEN,
            "threshold":  0.5,
        }
        with open(MODEL_DIR / "lstm_config.json", "w") as f:
            json.dump(config, f, indent=2)

        log.info(f"LSTM config saved → {MODEL_DIR / 'lstm_config.json'}")

    return model


if __name__ == "__main__":
    train()