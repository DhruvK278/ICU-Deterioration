# ICU Patient Deterioration Prediction
**Edge-Fog-Cloud ML System | Healthcare Domain**

Predicts ICU patient deterioration 6–12 hours in advance using a three-tier distributed ML architecture.

---

## Project structure
```
icu-deterioration/
├── data/
│   ├── raw/            ← place MIMIC CSV here (gitignored)
│   └── processed/      ← generated parquet splits (gitignored)
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── edge/           ← lightweight real-time anomaly detector
│   ├── fog/            ← FastAPI LSTM inference server
│   └── cloud/          ← training scripts, data pipeline
├── docker/             ← Dockerfiles per tier
├── dashboard/          ← Streamlit nurse alert UI
├── mlflow/             ← experiment config
└── .github/workflows/  ← CI/CD
```

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/DhruvK278/ICU-Deterioration.git
cd icu-deterioration
pip install -r requirements.txt

# 2. Get data (option A — real MIMIC)
#    Download from https://www.kaggle.com/datasets/drscarlat/mimic3d
#    Place the CSV in data/raw/

# 3. Get data (option B — synthetic for testing)
python src/cloud/generate_synthetic.py

# 4. Run the data pipeline
python src/cloud/data_pipeline.py

# 5. Explore in the notebook
jupyter notebook notebooks/01_eda.ipynb
```

## Dataset
- **Source**: MIMIC-III Clinical Database (simplified Kaggle version)
- **Link**: https://www.kaggle.com/datasets/drscarlat/mimic3d
- **Target**: `mort_icu` — ICU mortality (binary classification)
- **Features**: 40+ vital signs, lab values, and engineered clinical features
