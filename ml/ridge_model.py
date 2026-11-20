"""SPWS - machine-learning core (no web code).

Algorithm : Ridge regression (L2-regularised linear regression), closed-form solution
Inputs    : 7 in-semester features (see FEAT), standardised with z-scores
Output    : predicted final-exam mark, predicted total (/100) and a risk class
"""
import numpy as np

# Order of the input features (must match the order of the model coefficients)
FEAT = ["quiz1_lec", "quiz2_lec", "total_lecture",
        "quiz1_section", "quiz2_section", "total_section", "midterm"]


def ridge_train(X, y, alpha=1.0, lo=0.0, hi=40.0):
    """Fit a ridge model.  coef = (Z'Z + alpha*I)^-1 Z'(y - mean(y)),  Z = standardised X.

    Returns a plain dict (JSON-serialisable) holding everything needed for prediction:
    coef, intercept, mean, scale and the output limits [min, max].
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mean = X.mean(0)
    scale = X.std(0)
    scale[scale == 0] = 1                      # avoid division by zero for constant columns
    Z = (X - mean) / scale                     # z-score standardisation
    yc = y - y.mean()                          # centre the target
    coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ yc)
    return dict(coef=coef.tolist(), intercept=float(y.mean()),
                mean=mean.tolist(), scale=scale.tolist(), min=lo, max=hi)


def ridge_predict(m, X):
    """Predict the final-exam mark for one or many students (X = rows of 7 features)."""
    z = (np.asarray(X, float) - m["mean"]) / m["scale"]
    return np.clip(z @ np.asarray(m["coef"]) + m["intercept"], m["min"], m["max"])


def predicted_total(midterm, total_lecture, total_section, pred_final):
    """Predicted total mark out of 100 = known marks + predicted final exam."""
    return min(100.0, max(0.0, midterm + total_lecture + total_section + float(pred_final)))


def risk(total):
    """Rule-based risk class from the predicted total (thresholds 50 and 65)."""
    return "High Risk" if total < 50 else "Medium Risk" if total < 65 else "Low Risk"


def predict_student(model, features):
    """Convenience wrapper: features = list of 7 numbers in FEAT order."""
    pf = float(ridge_predict(model, [features])[0])
    pt = predicted_total(features[6], features[2], features[5], pf)
    return dict(pred_final=round(pf, 1), pred_total=round(pt, 1), risk=risk(pt))
