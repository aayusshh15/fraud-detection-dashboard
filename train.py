"""Train, evaluate and export everything the dashboard needs into artifacts/.

Usage:
    python train.py                      # uses creditcard.csv
    python train.py --data path/to.csv
    python train.py --synthetic --fast   # quick smoke test, no dataset needed
"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, precision_recall_curve)
warnings.filterwarnings("ignore")
RS = 42


def load(path, synthetic):
    if not synthetic:
        return pd.read_csv(path), "real"
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=30000, n_features=28, n_informative=12,
                               weights=[0.99, 0.01], class_sep=1.2, flip_y=0.002, random_state=RS)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    rng = np.random.default_rng(RS)
    df["Time"] = np.sort(rng.integers(0, 172800, len(df)))
    df["Amount"] = np.round(rng.lognormal(3.5, 1.2, len(df)), 2)
    df["Class"] = y
    return df[["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]], "synthetic"


def evaluate(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return dict(Precision=precision_score(y, pred, zero_division=0), Recall=recall_score(y, pred),
                F1=f1_score(y, pred), ROC_AUC=roc_auc_score(y, proba), PR_AUC=average_precision_score(y, proba))


def main(a):
    out = Path(a.out); out.mkdir(exist_ok=True)
    df, source = load(a.data, a.synthetic)
    df = df.drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns="Class"), df["Class"]
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=RS)
    prep = ColumnTransformer([("sc", RobustScaler(), ["Time", "Amount"])], remainder="passthrough")
    make = lambda c: Pipeline([("prep", prep), ("clf", c)])
    n = 60 if a.fast else 200
    models = {
        "Baseline (majority)": make(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": make(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RS)),
        "Decision Tree": make(DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=RS)),
        "Random Forest": make(RandomForestClassifier(n_estimators=n, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=RS)),
        "Gradient Boosting": make(HistGradientBoostingClassifier(class_weight="balanced", max_iter=n + 100, random_state=RS)),
        "Voting (LR+RF+GB)": make(VotingClassifier([
            ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RS)),
            ("rf", RandomForestClassifier(n_estimators=n, class_weight="balanced_subsample", n_jobs=-1, random_state=RS)),
            ("gb", HistGradientBoostingClassifier(class_weight="balanced", max_iter=n + 100, random_state=RS))], voting="soft")),
    }
    val_rows, fitted = [], {}
    for name, m in models.items():
        t0 = time.time(); m.fit(X_tr, y_tr); sec = time.time() - t0
        val_rows.append({"Model": name, **evaluate(y_va, m.predict_proba(X_va)[:, 1]), "Train_sec": sec})
        fitted[name] = m; print(f"trained {name} ({sec:.1f}s)")
    val = pd.DataFrame(val_rows).set_index("Model").round(4)
    best = val.drop(index="Baseline (majority)")["PR_AUC"].idxmax()
    p, r, t = precision_recall_curve(y_va, fitted[best].predict_proba(X_va)[:, 1])
    thr = float(t[(2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)).argmax()])

    scores = pd.DataFrame({"y_true": y_te.values, "amount": X_te["Amount"].values})
    test_rows = []
    for name, m in fitted.items():
        s = m.predict_proba(X_te)[:, 1]; scores[name] = s
        test_rows.append({"Model": name, **evaluate(y_te, s, thr if name == best else 0.5)})
    test = pd.DataFrame(test_rows).set_index("Model").round(4)
    tn, fp, fn, tp = confusion_matrix(y_te, (scores[best] >= thr).astype(int)).ravel()

    rf = fitted["Random Forest"].named_steps["clf"]
    cols = ["Time", "Amount"] + [c for c in X.columns if c not in ("Time", "Amount")]
    fi = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)

    sample = pd.concat([X_te[y_te == 1], X_te[y_te == 0].sample(min(1500, (y_te == 0).sum()), random_state=RS)])
    sample = sample.assign(Class=y_te.loc[sample.index]).sample(frac=1, random_state=RS)

    val.to_csv(out / "comparison_val.csv"); test.to_csv(out / "comparison_test.csv")
    scores.round(5).to_csv(out / "test_scores.csv", index=False)
    sample.to_csv(out / "test_sample.csv", index=False)
    fi.rename("importance").to_csv(out / "feature_importance.csv")
    joblib.dump({"model": fitted[best], "threshold": thr, "columns": list(X.columns), "name": best}, out / "model.joblib")
    json.dump({"source": source, "rows": len(df), "fraud_rows": int(y.sum()), "fraud_rate": float(y.mean()),
               "split": {"train": len(X_tr), "val": len(X_va), "test": len(X_te)}, "best_model": best,
               "threshold": thr, "test_metrics": test.loc[best].to_dict(),
               "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}},
              open(out / "metrics.json", "w"), indent=2)
    print(f"\nBest: {best} | threshold {thr:.3f} | data: {source}\n{test}\nArtifacts saved in {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="creditcard.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fast", action="store_true")
    main(ap.parse_args())
