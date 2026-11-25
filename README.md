**Live demo:** https://fraud-detection-dashboard-7nnj3jwnyfdcw7upzszzg7.streamlit.app/

# 🛡️ Credit Card Fraud Detection: ML Dashboard

End-to-end machine learning project: imbalanced classification with ensemble models, a cost-aware decision threshold,
and an interactive Streamlit dashboard. 

> Decision-support demo only. Not validated for real financial decisions.

## What it does
- **Trains and compares 6 models**: majority baseline, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, soft Voting.
- **Handles ~0.17% class imbalance** with class weights (no resampling before the split), stratified 60/20/20 split, scaling inside a Pipeline (no leakage).
- **Ranks models by PR-AUC**, tunes the threshold on validation data, reports on the test set once.
- **Dashboard** with 5 tabs: overview, model comparison (PR/ROC curves), threshold & business-cost tuning, transaction scoring (built-in sample or CSV upload, with download), insights & ethics.

## Quick start
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# download creditcard.csv from Kaggle ("Credit Card Fraud Detection", ULB) into this folder
python train.py            # trains and writes artifacts/
streamlit run app.py       # opens the dashboard at http://localhost:8501
python -m pytest           # optional smoke test (uses synthetic data)
```
No dataset yet? `python train.py --synthetic --fast` creates demo artifacts (the dashboard shows a warning banner).

## Project structure
| File | Purpose |
|---|---|
| `train.py` | Preprocessing, training, evaluation, exports to `artifacts/` |
| `app.py` | Streamlit dashboard |
| `fraud_detection.ipynb` | Step-by-step notebook version with EDA and error analysis |
| `tests/test_smoke.py` | End-to-end test of the training pipeline |
| `artifacts/` | Generated model, metrics and scores (created by `train.py`) |

## Results
Add your real numbers here after running `train.py` (copy from `artifacts/comparison_test.csv`), plus 2-3 dashboard screenshots in a `docs/` folder.

## Data & ethics
Kaggle/ULB dataset: 284,807 transactions (Sept 2013, European cardholders), 492 frauds; features V1-V28 are PCA-anonymised.
Fraud patterns drift over time, false alarms affect real customers, and every flag should be reviewed by a human.
