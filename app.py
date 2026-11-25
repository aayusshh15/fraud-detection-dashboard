import json
from pathlib import Path
import joblib, numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

ART = Path("artifacts")
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")

if not (ART / "model.joblib").exists():
    st.title("🛡️ Fraud Detection Dashboard")
    st.error("No trained artifacts found. Run `python train.py` first (needs creditcard.csv), then restart.")
    st.stop()


@st.cache_data
def load_tables():
    return (json.load(open(ART / "metrics.json")), pd.read_csv(ART / "comparison_val.csv", index_col=0),
            pd.read_csv(ART / "comparison_test.csv", index_col=0), pd.read_csv(ART / "test_scores.csv"),
            pd.read_csv(ART / "feature_importance.csv", index_col=0)["importance"])


@st.cache_resource
def load_model():
    return joblib.load(ART / "model.joblib")


meta, val, test, scores, fi = load_tables()
art = load_model(); best = meta["best_model"]
model_names = [c for c in scores.columns if c not in ("y_true", "amount")]

st.title("🛡️ Credit Card Fraud Detection")
st.caption("Decision-support demo that ranks transactions for human review. It is not proof of fraud. "
           + ("⚠️ Showing SYNTHETIC data." if meta["source"] == "synthetic" else ""))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🏁 Model comparison", "⚖️ Threshold & cost",
                                        "🔍 Score transactions", "🧠 Insights & ethics"])

with tab1:
    m, c = meta["test_metrics"], meta["confusion"]
    a, b, c3, d = st.columns(4)
    a.metric("Best model", best); b.metric("PR-AUC (test)", f"{m['PR_AUC']:.3f}")
    c3.metric("Recall (frauds caught)", f"{m['Recall']:.1%}"); d.metric("Precision (flags that are fraud)", f"{m['Precision']:.1%}")
    l, r = st.columns(2)
    with l:
        st.subheader("Dataset")
        st.write(f"**{meta['rows']:,}** transactions, **{meta['fraud_rows']:,}** frauds (**{meta['fraud_rate']:.3%}**).  \n"
                 f"Split (stratified): train {meta['split']['train']:,} / val {meta['split']['val']:,} / test {meta['split']['test']:,}.")
        st.write("Accuracy is misleading here: predicting 'legit' every time scores ~99.8%. "
                 "Models are ranked by **PR-AUC**, and the decision threshold was tuned on validation data only.")
    with r:
        st.subheader("Confusion matrix (test set)")
        fig, ax = plt.subplots(figsize=(3.6, 3))
        cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]]); ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm): ax.text(j, i, f"{v:,}", ha="center", va="center")
        ax.set_xticks([0, 1], ["Legit", "Fraud"]); ax.set_yticks([0, 1], ["Legit", "Fraud"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); st.pyplot(fig)

with tab2:
    st.subheader("Test-set comparison")
    st.dataframe(test.style.highlight_max(axis=0, color="#cfe8cf"), use_container_width=True)
    l, r = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for n in model_names:
            if n.startswith("Baseline"): continue
            p, rc, _ = precision_recall_curve(scores.y_true, scores[n]); ax.plot(rc, p, label=n)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall curves"); ax.legend(fontsize=7)
        st.pyplot(fig)
    with r:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for n in model_names:
            fpr, tpr, _ = roc_curve(scores.y_true, scores[n]); ax.plot(fpr, tpr, label=n)
        ax.plot([0, 1], [0, 1], "k--", lw=.7); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves"); st.pyplot(fig)
    st.subheader("Training time (s)")
    st.bar_chart(val["Train_sec"])
    st.caption("Ensembles (Random Forest, Gradient Boosting, Voting) are compared against single models and a majority baseline.")

with tab3:
    st.subheader("Pick a threshold by business cost")
    c1, c2, c3 = st.columns(3)
    thr = c1.slider("Decision threshold", 0.01, 0.99, float(meta["threshold"]), 0.01)
    review = c2.number_input("Cost of reviewing one flagged transaction", 0.0, 1000.0, 5.0, 1.0)
    miss_mult = c3.number_input("Missed fraud costs this × its amount", 0.0, 10.0, 1.0, 0.1)
    s, y, amt = scores[best].values, scores.y_true.values, scores.amount.values
    ths = np.linspace(0.01, 0.99, 99)
    def cost(t):
        f = s >= t
        return review * f.sum() + miss_mult * amt[(y == 1) & ~f].sum()
    costs = np.array([cost(t) for t in ths]); t_best = ths[costs.argmin()]
    f = s >= thr; tp = int(((y == 1) & f).sum()); fp = int(((y == 0) & f).sum()); fn = int(((y == 1) & ~f).sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Flagged", f"{int(f.sum()):,}"); k2.metric("Frauds caught", f"{tp} / {tp + fn}")
    k3.metric("False alarms", f"{fp:,}"); k4.metric("Total cost at threshold", f"{cost(thr):,.0f}")
    fig, ax = plt.subplots(figsize=(7, 3)); ax.plot(ths, costs); ax.axvline(thr, color="r", ls="--", label="your threshold")
    ax.axvline(t_best, color="g", ls=":", label=f"cheapest ({t_best:.2f})"); ax.set_xlabel("Threshold"); ax.set_ylabel("Cost"); ax.legend()
    st.pyplot(fig)
    st.caption("Lower threshold: more fraud caught, more false alarms. Higher: fewer reviews, more fraud slips through.")

with tab4:
    st.subheader("Score transactions")
    src = st.radio("Input", ["Built-in test sample", "Upload CSV"], horizontal=True)
    df = None
    if src == "Built-in test sample":
        df = pd.read_csv(ART / "test_sample.csv")
    else:
        up = st.file_uploader("CSV with columns Time, V1-V28, Amount (Class optional)", type="csv")
        if up: df = pd.read_csv(up)
    if df is not None:
        miss = [c for c in art["columns"] if c not in df.columns]
        if miss:
            st.error(f"Missing columns: {miss}")
        else:
            t_use = st.slider("Threshold", 0.01, 0.99, float(art["threshold"]), 0.01, key="t4")
            out = df.copy(); out["fraud_score"] = art["model"].predict_proba(df[art["columns"]])[:, 1]
            out["flagged"] = (out.fraud_score >= t_use).astype(int)
            a, b, c = st.columns(3)
            a.metric("Transactions", f"{len(out):,}"); b.metric("Flagged", int(out.flagged.sum())); c.metric("Flag rate", f"{out.flagged.mean():.2%}")
            if "Class" in out:
                st.write(f"Of the **{int(out.Class.sum())}** real frauds in this file, **{int(((out.Class == 1) & (out.flagged == 1)).sum())}** were flagged.")
            top = out.sort_values("fraud_score", ascending=False)
            st.dataframe(top.head(50), use_container_width=True)
            st.download_button("Download scored CSV", top.to_csv(index=False), "scored_transactions.csv")
            st.markdown("**Inspect one transaction**")
            i = st.number_input("Row index in table above (0 = riskiest)", 0, len(top) - 1, 0)
            row = top.iloc[int(i)]; st.progress(float(row.fraud_score), text=f"Fraud score {row.fraud_score:.3f} → {'FLAG for review' if row.fraud_score >= t_use else 'looks normal'}")
            st.bar_chart(row[fi.index[:8]].astype(float), height=200)
            st.caption("Values of the 8 features the Random Forest relies on most, for this transaction.")

with tab5:
    l, r = st.columns(2)
    with l:
        st.subheader("Top features (Random Forest)"); st.bar_chart(fi.head(12))
        st.caption("V1-V28 are PCA components (anonymised), so features show *which* signals matter, not their real-world meaning.")
    with r:
        st.subheader("Limitations & responsible use")
        st.markdown("- Data are anonymised European card transactions from 2013; fraud patterns drift, so models need retraining.\n"
                    "- Only ~0.17% of rows are fraud, so metrics on the test set have wide uncertainty.\n"
                    "- False positives inconvenience real customers; a human should review every flag.\n"
                    "- Not validated for real financial decisions. Educational demonstration only.\n"
                    "- No personal data is stored by this app; uploads are processed in memory.")
