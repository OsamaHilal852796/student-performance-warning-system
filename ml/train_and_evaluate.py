"""Train and evaluate the SPWS ridge model.

Usage
  python train_and_evaluate.py seed.json            # evaluate the stored models/predictions
  python train_and_evaluate.py students.xlsx        # train from a spreadsheet, then evaluate
                                                    # (columns: quiz1_lec ... midterm, final, final_total)
  python train_and_evaluate.py students.xlsx --holdout 0.2   # hold-out evaluation (unseen students)
"""
import sys, json
import numpy as np
from ridge_model import FEAT, ridge_train, ridge_predict, risk


def metrics(af, pf, at, pt):
    err = af - pf
    fail, high = at < 50, pt < 50
    tp, fn = int((fail & high).sum()), int((fail & ~high).sum())
    fp, tn = int((~fail & high).sum()), int((~fail & ~high).sum())
    return dict(
        n=len(af),
        MAE=round(float(np.abs(err).mean()), 2),
        baseline_MAE=round(float(np.abs(af - af.mean()).mean()), 2),
        RMSE=round(float(np.sqrt((err ** 2).mean())), 2),
        R2=round(float(1 - (err ** 2).sum() / ((af - af.mean()) ** 2).sum()), 3),
        within_3_marks_pct=round(float((np.abs(err) <= 3).mean() * 100), 1),
        confusion=dict(TP=tp, FN=fn, FP=fp, TN=tn),
        accuracy_pct=round(100 * (tp + tn) / len(af), 1),
        recall_pct=round(100 * tp / (tp + fn), 1) if tp + fn else None,
        precision_pct=round(100 * tp / (tp + fp), 1) if tp + fp else None,
    )


def evaluate_seed(path):
    d = json.load(open(path, encoding="utf8"))
    for s in d["models"]:
        g = [p for p in d["predictions"] if p["subject"] == s]
        a = lambda k: np.array([p[k] for p in g], float)
        print(s, json.dumps(metrics(a("actual_final"), a("pred_final"), a("actual_total"), a("pred_total")), indent=2))


def evaluate_table(path, holdout=0.0):
    import pandas as pd
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    X, y, tot = df[FEAT].apply(pd.to_numeric, errors="coerce").fillna(0).values, df["final"].astype(float).values, df["final_total"].astype(float).values
    idx = np.random.default_rng(42).permutation(len(df))
    cut = int(len(df) * (1 - holdout))
    tr, te = idx[:cut], idx[cut:] if holdout else idx
    model = ridge_train(X[tr], y[tr])
    pf = ridge_predict(model, X[te])
    pt = np.clip(X[te][:, 6] + X[te][:, 2] + X[te][:, 5] + pf, 0, 100)
    print("train rows:", len(tr), " test rows:", len(te), "(hold-out)" if holdout else "(in-sample)")
    print(json.dumps(metrics(y[te], pf, tot[te], pt), indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    path = sys.argv[1]
    ho = float(sys.argv[sys.argv.index("--holdout") + 1]) if "--holdout" in sys.argv else 0.0
    evaluate_seed(path) if path.endswith(".json") else evaluate_table(path, ho)
