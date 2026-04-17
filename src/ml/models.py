"""
ML models for RetainSight.

1. Churn Prediction — Random Forest + Logistic Regression with class balancing
2. Customer Segmentation — KMeans clustering on behavioral features
3. LTV Estimation — Gradient Boosting Regressor
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .feature_engineering import build_feature_matrix

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


def _ensure_model_dir() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR


# ---------------------------------------------------------------------------
# 1. Churn Prediction
# ---------------------------------------------------------------------------

def train_churn_model() -> dict:
    """
    Train churn classifiers and return evaluation metrics.
    Saves the best model + scaler to disk.
    """
    df = build_feature_matrix()
    feature_cols = [c for c in df.columns if c not in ("customer_id", "is_churned")]
    X = df[feature_cols].values
    y = df["is_churned"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    lr.fit(X_train_s, y_train)

    results = {}
    for name, model in [("RandomForest", rf), ("LogisticRegression", lr)]:
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]
        results[name] = {
            "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }

    best_name = max(results, key=lambda k: results[k]["auc_roc"])
    best_model = rf if best_name == "RandomForest" else lr

    importances = None
    if best_name == "RandomForest":
        importances = dict(zip(feature_cols, rf.feature_importances_))

    out = _ensure_model_dir()
    with open(out / "churn_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(out / "churn_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out / "churn_features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    return {
        "best_model": best_name,
        "results": results,
        "feature_importances": importances,
    }


def predict_churn() -> pd.DataFrame:
    """
    Load trained model and score every customer.
    Returns DataFrame with customer_id, churn_probability, risk_tier.
    """
    out = _ensure_model_dir()
    with open(out / "churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(out / "churn_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(out / "churn_features.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    df = build_feature_matrix()
    X = df[feature_cols].values
    X_s = scaler.transform(X)

    proba = model.predict_proba(X_s)[:, 1]
    df["churn_probability"] = np.round(proba, 4)
    df["risk_tier"] = pd.cut(
        proba,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )

    return df[["customer_id", "churn_probability", "risk_tier", "is_churned",
               "total_spend", "current_mrr", "tenure_days",
               "events_last_30d", "days_since_last_event"]]


# ---------------------------------------------------------------------------
# 2. Customer Segmentation
# ---------------------------------------------------------------------------

def train_segmentation(n_clusters: int = 4) -> dict:
    """
    KMeans segmentation on behavioral + value features.
    Returns cluster profiles.
    """
    df = build_feature_matrix()
    seg_features = [
        "tenure_days", "total_spend", "num_transactions",
        "total_events", "login_count", "feature_use_count",
        "events_last_30d", "current_mrr",
    ]
    X = df[seg_features].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_s)

    sil = silhouette_score(X_s, labels)

    df["segment"] = labels
    profiles = df.groupby("segment").agg(
        size=("customer_id", "count"),
        avg_spend=("total_spend", "mean"),
        avg_events=("total_events", "mean"),
        avg_mrr=("current_mrr", "mean"),
        churn_rate=("is_churned", "mean"),
        avg_tenure=("tenure_days", "mean"),
    ).round(2)

    segment_names = []
    for _, row in profiles.iterrows():
        if row["avg_spend"] > profiles["avg_spend"].median() and row["churn_rate"] < 0.2:
            segment_names.append("High-Value Loyal")
        elif row["avg_spend"] > profiles["avg_spend"].median() and row["churn_rate"] >= 0.2:
            segment_names.append("High-Value At Risk")
        elif row["avg_spend"] <= profiles["avg_spend"].median() and row["churn_rate"] < 0.2:
            segment_names.append("Growth Potential")
        else:
            segment_names.append("Low-Value Churning")
    profiles["segment_name"] = segment_names

    out = _ensure_model_dir()
    with open(out / "segmentation_model.pkl", "wb") as f:
        pickle.dump({"kmeans": km, "scaler": scaler, "features": seg_features}, f)

    return {
        "silhouette_score": round(sil, 4),
        "profiles": profiles.reset_index(),
    }


# ---------------------------------------------------------------------------
# 3. LTV Estimation
# ---------------------------------------------------------------------------

def train_ltv_model() -> dict:
    """
    Train a Gradient Boosting Regressor to predict customer LTV
    (approximated as total_spend).
    """
    df = build_feature_matrix()
    ltv_features = [
        "tenure_days", "age", "num_transactions",
        "total_events", "login_count", "feature_use_count",
        "current_mrr", "plan_encoded", "events_last_30d",
    ]

    X = df[ltv_features].values
    y = df["total_spend"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    out = _ensure_model_dir()
    with open(out / "ltv_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": ltv_features}, f)

    importances = dict(zip(ltv_features, model.feature_importances_))

    return {
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "feature_importances": importances,
    }
