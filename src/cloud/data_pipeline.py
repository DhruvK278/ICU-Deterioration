"""
src/cloud/data_pipeline.py

MIMIC-III ICU Deterioration Prediction
Data loading, cleaning, feature engineering, and train/val/test splitting.

Dataset: https://www.kaggle.com/datasets/drscarlat/mimic3d
Actual columns: hadm_id, gender, age, LOSdays, admit_type, admit_location,
  AdmitDiagnosis, insurance, religion, marital_status, ethnicity,
  NumCallouts, NumDiagnosis, NumProcs, AdmitProcedure, NumCPTevents,
  NumInput, NumLabs, NumMicroLabs, NumNotes, NumOutput, NumRx,
  NumProcEvents, NumTransfers, NumChartEvents, ExpiredHospital,
  TotalNumInteract, LOSgroupNum
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[2]
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "expiredhospital"

COUNT_COLS = [
    "numcallouts", "numdiagnosis", "numprocs", "numcptevents",
    "numinput", "numlabs", "numicrolabs", "numnotes", "numoutput",
    "numrx", "numprocevents", "numtransfers", "numchartevents",
    "totalNumInteract",
]

DEMOGRAPHIC_COLS = ["age", "gender"]

# procedure fields
SEPSIS_KW    = ["sepsis", "septic", "bacteremia", "bacteraemia"]
CARDIAC_KW   = ["cardiac arrest", "heart failure", "myocardial", "cardiogenic"]
RESPIRATORY_KW = ["respiratory failure", "pneumonia", "ards", "intubat"]
TRAUMA_KW    = ["trauma", "hemorrhage", "haemorrhage", "bleeding"]

CATEGORICAL_COLS = [
    "admit_type", "admit_location", "insurance",
    "marital_status", "ethnicity",
]


# 1. Load 
def load_raw(path: Path = None) -> pd.DataFrame:
    if path is None:
        candidates = list(RAW_DIR.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No CSV files found in {RAW_DIR}.\n"
                "Download from https://www.kaggle.com/datasets/drscarlat/mimic3d "
                "and place the CSV in data/raw/"
            )
        path = candidates[0]

    log.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    log.info(f"Raw shape: {df.shape}  |  columns: {df.columns.tolist()}")
    return df


# 2. Clean
def clean(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    before = len(df)
    df = df.drop_duplicates(subset=["hadm_id"] if "hadm_id" in df.columns else None)
    log.info(f"Dropped {before - len(df)} duplicate admissions")

    if "losdays" in df.columns:
        df["losdays"] = df["losdays"].clip(0, 365)

    if "age" in df.columns:
        df["age"] = df["age"].clip(0, 90)

    if "ethnicity" in df.columns:
        eth = df["ethnicity"].str.upper().fillna("UNKNOWN")
        df["ethnicity"] = np.select(
            [
                eth.str.contains("WHITE"),
                eth.str.contains("BLACK|AFRICAN|CAPE VERDEAN|HAITIAN"),
                eth.str.contains("HISPANIC|LATINO|SOUTH AMERICAN|PORTUGUESE"),
                eth.str.contains("ASIAN|CAMBODIAN|CHINESE|FILIPINO|JAPANESE|KOREAN|THAI|VIETNAMESE"),
                eth.str.contains("MIDDLE EASTERN|NATIVE HAWAIIAN|PACIFIC|CARIBBEAN|MULTI RACE|AMERICAN INDIAN"),
            ],
            ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"],
            default="UNKNOWN"
        )

    log.info(f"Cleaned shape: {df.shape}")
    return df


# 3. Feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "numlabs" in df.columns and "numchartevents" in df.columns:
        df["acuity_score"] = (
            df.get("numlabs", 0).fillna(0) * 1.5 +
            df.get("numchartevents", 0).fillna(0) * 1.0 +
            df.get("numprocevents", 0).fillna(0) * 2.0 +
            df.get("numinput", 0).fillna(0) * 1.2 +
            df.get("numoutput", 0).fillna(0) * 1.2 +
            df.get("numrx", 0).fillna(0) * 1.0
        )

    if "numinput" in df.columns and "numoutput" in df.columns:
        total_io = df["numinput"].fillna(0) + df["numoutput"].fillna(0)
        df["io_ratio"] = df["numinput"].fillna(0) / total_io.replace(0, np.nan)
    if "numicrolabs" in df.columns and "numlabs" in df.columns:
        df["micro_rate"] = df["numicrolabs"].fillna(0) / df["numlabs"].replace(0, np.nan)

    if "numtransfers" in df.columns:
        df["high_transfer"] = (df["numtransfers"] > 2).astype(int)

    if "losdays" in df.columns:
        df["los_gt7"]  = (df["losdays"] > 7).astype(int)
        df["los_gt14"] = (df["losdays"] > 14).astype(int)
        df["los_gt30"] = (df["losdays"] > 30).astype(int)

    if "age" in df.columns:
        df["age_gt65"] = (df["age"] > 65).astype(int)
        df["age_gt80"] = (df["age"] > 80).astype(int)

    for col in ["admitdiagnosis", "admitprocedure"]:
        if col not in df.columns:
            continue
        text = df[col].fillna("").str.lower()
        df["dx_sepsis"]      = df.get("dx_sepsis",      pd.Series(0, index=df.index)) | text.str.contains("|".join(SEPSIS_KW),      regex=True).astype(int)
        df["dx_cardiac"]     = df.get("dx_cardiac",     pd.Series(0, index=df.index)) | text.str.contains("|".join(CARDIAC_KW),     regex=True).astype(int)
        df["dx_respiratory"] = df.get("dx_respiratory", pd.Series(0, index=df.index)) | text.str.contains("|".join(RESPIRATORY_KW), regex=True).astype(int)
        df["dx_trauma"]      = df.get("dx_trauma",      pd.Series(0, index=df.index)) | text.str.contains("|".join(TRAUMA_KW),      regex=True).astype(int)

    if "numcallouts" in df.columns:
        df["has_callouts"] = (df["numcallouts"] > 0).astype(int)

    log.info(f"Features after engineering: {df.shape[1]} columns")
    return df


# 4. Encode categoricals
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "gender" in df.columns:
        df["gender"] = (df["gender"].str.upper() == "M").astype(int)

    cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    if cols_present:
        df = pd.get_dummies(df, columns=cols_present, drop_first=True, dtype=int)

    return df


# 5. Select features & impute
ENGINEERED_COLS = [
    "acuity_score", "io_ratio", "micro_rate",
    "high_transfer", "los_gt7", "los_gt14", "los_gt30",
    "age_gt65", "age_gt80",
    "dx_sepsis", "dx_cardiac", "dx_respiratory", "dx_trauma",
    "has_callouts",
]

def select_and_impute(df: pd.DataFrame):
    base_cols = (
        [c for c in COUNT_COLS       if c in df.columns] +
        [c for c in DEMOGRAPHIC_COLS if c in df.columns] +
        ["losdays"] if "losdays" in df.columns else [] +
        [c for c in ENGINEERED_COLS  if c in df.columns]
    )
    ohe_cols = [c for c in df.columns if any(
        c.startswith(cat + "_") for cat in CATEGORICAL_COLS
    )]
    all_feature_cols = list(dict.fromkeys(base_cols + ohe_cols))  # dedupe, preserve order

    X = df[all_feature_cols].copy()
    y = df[TARGET_COL].copy()

    missing_pct = X.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 50].index.tolist()
    if high_missing:
        log.warning(f"Dropping columns with >50% missing: {high_missing}")
        X = X.drop(columns=high_missing)

    X = X.fillna(X.median(numeric_only=True))

    log.info(f"Feature matrix: {X.shape}  |  target distribution: {y.value_counts().to_dict()}")
    return X, y, X.columns.tolist()


# 6. Split & scale
def split_and_scale(X: pd.DataFrame, y: pd.Series):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_sc   = pd.DataFrame(scaler.transform(X_val),       columns=X_val.columns,   index=X_val.index)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,  index=X_test.index)

    log.info(f"Split — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    log.info(f"Class balance (train) — survived: {(y_train==0).sum()}, died: {(y_train==1).sum()}")
    return X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, scaler


# 7. Save
def save_processed(X_train, X_val, X_test, y_train, y_val, y_test, scaler):
    X_train.to_parquet(PROC_DIR / "X_train.parquet")
    X_val.to_parquet(PROC_DIR   / "X_val.parquet")
    X_test.to_parquet(PROC_DIR  / "X_test.parquet")
    y_train.to_frame().to_parquet(PROC_DIR / "y_train.parquet")
    y_val.to_frame().to_parquet(PROC_DIR   / "y_val.parquet")
    y_test.to_frame().to_parquet(PROC_DIR  / "y_test.parquet")
    joblib.dump(scaler, PROC_DIR / "scaler.pkl")
    log.info(f"Saved processed splits + scaler → {PROC_DIR}")


# 8. Full pipeline
def run_pipeline(raw_path: Path = None):
    df = load_raw(raw_path)
    df = clean(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    X, y, features = select_and_impute(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X, y)
    save_processed(X_train, X_val, X_test, y_train, y_val, y_test, scaler)

    summary = {
        "total_samples": len(df),
        "features":      len(features),
        "feature_names": features,
        "train_size":    len(X_train),
        "val_size":      len(X_val),
        "test_size":     len(X_test),
        "positive_rate": float(y.mean().round(4)),
    }
    log.info(f"Pipeline complete: {summary}")
    return summary


if __name__ == "__main__":
    run_pipeline()