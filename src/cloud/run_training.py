"""
src/cloud/run_training.py

Runs both XGBoost and LSTM training in sequence and prints a final
comparison. Also starts MLflow UI instructions.

Usage:
    python src/cloud/run_training.py
    python src/cloud/run_training.py --model xgb     # XGBoost only
    python src/cloud/run_training.py --model lstm    # LSTM only
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def run_xgb():
    log.info("=" * 60)
    log.info("PHASE 2A — XGBoost baseline")
    log.info("=" * 60)
    from train_xgboost import train as xgb_train
    model, importance = xgb_train()
    return model


def run_lstm():
    log.info("=" * 60)
    log.info("PHASE 2B — LSTM fog model")
    log.info("=" * 60)
    from train_lstm import train as lstm_train
    model = lstm_train()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgb", "lstm", "both"], default="both")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src" / "cloud"))

    if args.model in ("xgb", "both"):
        run_xgb()

    if args.model in ("lstm", "both"):
        run_lstm()

    log.info("\n" + "=" * 60)
    log.info("Training complete. View results:")
    log.info("  mlflow ui --host 0.0.0.0 --port 5000")
    log.info("  then open http://localhost:5000")
    log.info("=" * 60)
    log.info("\nModels saved to models/:")
    for f in (ROOT / "models").glob("*"):
        log.info(f"  {f.name}")