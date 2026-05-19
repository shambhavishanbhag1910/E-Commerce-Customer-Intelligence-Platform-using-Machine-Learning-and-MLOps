from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "delivery_delay_model.pkl"

app = FastAPI(
    title="E-Commerce Delivery Delay Prediction API",
    description="API to predict whether an e-commerce order may be delayed",
    version="1.0.0"
)


class OrderInput(BaseModel):
    customer_state: str
    product_category_name: str
    payment_type: str
    payment_value: float
    freight_value: float
    price: float
    product_weight_g: float
    product_volume: float
    purchase_dayofweek: int
    purchase_month: int
    purchase_year: int
    estimated_delivery_days: int


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/")
def home():
    return {
        "message": "E-Commerce Delivery Delay Prediction API is running"
    }


@app.post("/predict")
def predict_delay(order: OrderInput):
    input_df = pd.DataFrame([order.model_dump()])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "delay_probability": round(float(probability), 4),
        "prediction_label": "Delayed" if prediction == 1 else "On Time"
    }