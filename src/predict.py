from pathlib import Path
import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "delivery_delay_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    return model


def make_prediction(input_data):
    model = load_model()

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = {
        "prediction": int(prediction),
        "delay_probability": round(float(probability), 4),
        "prediction_label": "Delayed" if prediction == 1 else "On Time"
    }

    return result


if __name__ == "__main__":
    sample_order = {
        "customer_state": "SP",
        "product_category_name": "cool_stuff",
        "payment_type": "credit_card",
        "payment_value": 150.0,
        "freight_value": 20.0,
        "price": 130.0,
        "product_weight_g": 650.0,
        "product_volume": 12000.0,
        "purchase_dayofweek": 2,
        "purchase_month": 9,
        "purchase_year": 2017,
        "estimated_delivery_days": 12
    }

    result = make_prediction(sample_order)

    print("Prediction Result:")
    print(result)