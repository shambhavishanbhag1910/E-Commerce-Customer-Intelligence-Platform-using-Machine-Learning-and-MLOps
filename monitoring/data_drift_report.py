from pathlib import Path
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Dataset by Olist.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "monitoring" / "data_drift_report.html"


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

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

    return df[features]


def main():
    df = load_and_prepare_data()

    reference_data = df.sample(frac=0.5, random_state=42)
    current_data = df.drop(reference_data.index)

    report = Report(metrics=[
        DataDriftPreset()
    ])

    report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))

    if REPORT_PATH.exists():
        print(f"Data drift report saved successfully at: {REPORT_PATH}")
    else:
        raise FileNotFoundError(f"Report was not created at: {REPORT_PATH}")


if __name__ == "__main__":
    main()