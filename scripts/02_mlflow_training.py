"""
02_mlflow_training.py
=====================
XGBoost + SMOTE training with full MLflow experiment tracking.
Benchmarks: Logistic Regression, Random Forest, XGBoost (tuned via RandomizedSearchCV).
Adds SHAP explainability plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
MODEL_DIR   = ROOT / "models"
OUTPUT_DIR  = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

PROCESSED   = DATA_DIR / "processed_features.csv"
MODEL_PATH  = MODEL_DIR / "xgb_churn_model.joblib"
LEADERBOARD = OUTPUT_DIR / "leaderboard.csv"


def load_processed():
    if not PROCESSED.exists():
        raise FileNotFoundError("[!] Run 01_eda_preprocessing.py first.")
    df = pd.read_csv(PROCESSED)
    X  = df.drop(columns=["AttritionBinary"])
    y  = df["AttritionBinary"]
    print(f"[✓] Loaded: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"    Class dist — 0: {(y==0).sum()} | 1: {(y==1).sum()} ({y.mean():.2%})")
    return X, y


def apply_smote(X_train, y_train):
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[✓] SMOTE: {X_res.shape[0]} samples ({y_res.mean():.2%} positive)")
    return X_res, y_res


def evaluate(model, X_test, y_test, model_name):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    metrics = {
        "Model":     model_name,
        "AUC-ROC":   round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    return metrics, y_prob, y_pred


def tune_xgboost(X_train, y_train):
    print("  Running RandomizedSearchCV (40 iterations, 5-fold CV)...")
    param_dist = {
        "n_estimators":     [300, 500, 700, 1000],
        "max_depth":        [3, 4, 5, 6, 7],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1],
        "subsample":        [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5, 7],
        "gamma":            [0, 0.1, 0.2, 0.3],
        "reg_alpha":        [0, 0.01, 0.1, 1.0],
        "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
    }
    neg_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    base = XGBClassifier(scale_pos_weight=neg_pos, eval_metric="auc",
                         random_state=42, n_jobs=-1, verbosity=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(base, param_dist, n_iter=40, scoring="roc_auc",
                                cv=cv, random_state=42, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    print(f"  Best CV AUC-ROC: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


def plot_roc_curves(models_probs, y_test, path):
    plt.figure(figsize=(8, 6))
    colors = ["#2563EB", "#16A34A", "#DC2626"]
    for (name, y_prob), color in zip(models_probs, colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[✓] ROC curves → {path}")


def plot_confusion_matrix(y_test, y_pred, path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"])
    plt.title("Confusion Matrix — XGBoost (Tuned)")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[✓] Confusion matrix → {path}")


def plot_feature_importance(model, feature_names, path):
    importances = model.feature_importances_
    top_idx  = np.argsort(importances)[::-1][:15]
    top_feats = [feature_names[i] for i in top_idx]
    top_vals  = importances[top_idx]
    plt.figure(figsize=(10, 6))
    plt.barh(top_feats[::-1], top_vals[::-1], color="#2563EB")
    plt.xlabel("Feature Importance (Gain)")
    plt.title("Top 15 Feature Importances — XGBoost (Tuned)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    fi_df = pd.DataFrame({"Feature": top_feats[:10], "Importance": top_vals[:10]})
    fi_df.to_csv(OUTPUT_DIR / "feature_importances.csv", index=False)
    print(f"[✓] Feature importances → outputs/feature_importances.csv")
    return fi_df


def plot_shap(model, X_test_arr, feature_names):
    print("\n[→] Computing SHAP values (~30s)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_arr)
    X_test_df   = pd.DataFrame(X_test_arr, columns=feature_names)

    # Beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, show=False, max_display=15)
    plt.title("SHAP Feature Impact — XGBoost (Tuned)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] SHAP summary → outputs/shap_summary.png")

    # Bar
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar",
                      show=False, max_display=15)
    plt.title("SHAP Mean Absolute Impact — XGBoost (Tuned)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] SHAP bar → outputs/shap_bar.png")

    # CSV for dashboard
    shap_df = pd.DataFrame(
        np.abs(shap_values).mean(axis=0).reshape(1, -1), columns=feature_names
    ).T.rename(columns={0: "MeanAbsSHAP"}).sort_values("MeanAbsSHAP", ascending=False)
    shap_df.head(15).to_csv(OUTPUT_DIR / "shap_values.csv")
    print(f"[✓] SHAP values CSV → outputs/shap_values.csv")


def main():
    mlflow.set_tracking_uri(f"file://{ROOT}/mlflow_runs")
    mlflow.set_experiment("churn-prediction")
    print("[✓] MLflow URI:", mlflow.get_tracking_uri())

    X, y = load_processed()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_res, y_res = apply_smote(X_train, y_train)

    results = []; models_probs = []

    # Logistic Regression
    print("\n── Logistic Regression ──────────────────────────────────────")
    with mlflow.start_run(run_name="logistic_regression"):
        lr = Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=1000, random_state=42,
                                                   class_weight="balanced", C=1.0))])
        lr.fit(X_res, y_res)
        m, yp, _ = evaluate(lr, X_test, y_test, "Logistic Regression")
        mlflow.log_params({"C": 1.0, "smote": True})
        mlflow.log_metrics({k: v for k, v in m.items() if k != "Model"})
        mlflow.sklearn.log_model(lr, "logistic_regression")
        results.append(m); models_probs.append(("Logistic Regression", yp))
        print(f"    AUC-ROC: {m['AUC-ROC']}  |  F1: {m['F1']}")

    # Random Forest
    print("\n── Random Forest ────────────────────────────────────────────")
    rf_p = {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 4,
            "class_weight": "balanced", "random_state": 42, "n_jobs": -1}
    with mlflow.start_run(run_name="random_forest"):
        rf = RandomForestClassifier(**rf_p)
        rf.fit(X_res, y_res)
        m, yp, _ = evaluate(rf, X_test, y_test, "Random Forest")
        mlflow.log_params({**rf_p, "smote": True})
        mlflow.log_metrics({k: v for k, v in m.items() if k != "Model"})
        mlflow.sklearn.log_model(rf, "random_forest")
        results.append(m); models_probs.append(("Random Forest", yp))
        print(f"    AUC-ROC: {m['AUC-ROC']}  |  F1: {m['F1']}")

    # XGBoost tuned
    print("\n── XGBoost (Tuned) ──────────────────────────────────────────")
    with mlflow.start_run(run_name="xgboost_tuned"):
        xgb_tuned, best_params, best_cv = tune_xgboost(X_res, y_res)
        m, yp_xgb, yp_pred = evaluate(xgb_tuned, X_test, y_test, "XGBoost (Tuned)")
        mlflow.log_params({**best_params, "smote": True, "tuning": "RandomizedSearchCV_40iter"})
        mlflow.log_metrics({**{k: v for k, v in m.items() if k != "Model"},
                             "best_cv_auc": round(best_cv, 4)})
        mlflow.xgboost.log_model(xgb_tuned, "xgboost_tuned")
        results.append(m); models_probs.append(("XGBoost (Tuned)", yp_xgb))
        print(f"    AUC-ROC: {m['AUC-ROC']}  |  F1: {m['F1']}")

    # Leaderboard
    lb = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False)
    lb.to_csv(LEADERBOARD, index=False)
    print(f"\n[✓] Leaderboard:\n{lb.to_string(index=False)}")

    # Plots
    plot_roc_curves(models_probs, y_test, OUTPUT_DIR / "roc_curves.png")
    plot_confusion_matrix(y_test, yp_pred, OUTPUT_DIR / "confusion_matrix.png")
    plot_feature_importance(xgb_tuned, list(X.columns), OUTPUT_DIR / "feature_importance.png")
    plot_shap(xgb_tuned, X_test.values, list(X.columns))

    # Save predictions + model
    pd.DataFrame({"ActualAttrition": y_test.values,
                  "ChurnProbability": yp_xgb,
                  "PredictedChurn": yp_pred}).to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    joblib.dump({"model": xgb_tuned, "feature_names": list(X.columns)}, MODEL_PATH)
    print(f"[✓] Model saved → {MODEL_PATH}")

    best = lb.iloc[0]
    print(f"\n{'='*55}")
    print(f"  Best: {best['Model']}  |  AUC-ROC: {best['AUC-ROC']}")
    print(f"  MLflow UI: mlflow ui --backend-store-uri {ROOT}/mlflow_runs")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
