import json, subprocess, sys
from pathlib import Path

def test_training_runs_and_exports(tmp_path):
    out = tmp_path / "art"
    subprocess.run([sys.executable, "train.py", "--synthetic", "--fast", "--out", str(out)], check=True)
    for f in ["model.joblib", "metrics.json", "comparison_test.csv", "test_scores.csv", "test_sample.csv", "feature_importance.csv"]:
        assert (out / f).exists(), f
    m = json.load(open(out / "metrics.json"))
    assert m["test_metrics"]["PR_AUC"] > 0.2
    assert 0 < m["threshold"] < 1
