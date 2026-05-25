from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "Dataset by Olist.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "delivery_delay_model.pkl"

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
EXPERIMENT_NAME = "delivery-delay-prediction"


def load_data(path):
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def prepare_data(df):
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df[df["order_status"] == "delivered"].copy()

    df = df.dropna(subset=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ])

    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days

    df["delivery_delay_flag"] = np.where(df["delivery_delay_days"] > 0, 1, 0)

    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year

    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    if all(col in df.columns for col in ["product_length_cm", "product_height_cm", "product_width_cm"]):
        df["product_volume"] = (
            df["product_length_cm"] *
            df["product_height_cm"] *
            df["product_width_cm"]
        )

    return df


def train_model(df):
    features = [
        "customer_state",
        "product_category_name",
        "payment_type",
        "payment_value",
        "freight_value",
        "price",
        "product_weight_g",
        "product_volume",
        "purchase_dayofweek",
        "purchase_month",
        "purchase_year",
        "estimated_delivery_days"
    ]

    features = [col for col in features if col in df.columns]

    X = df[features]
    y = df["delivery_delay_flag"]
    groups = df["order_id"]

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    # MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("split_strategy", "GroupShuffleSplit")
        mlflow.log_param("target", "delivery_delay_flag")
        mlflow.log_param("features", ",".join(features))

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

    return model


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preparing data...")
    df = prepare_data(df)

    print("Training model...")
    model = train_model(df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved successfully at: {MODEL_PATH}")


if __name__ == "__main__":
    main()