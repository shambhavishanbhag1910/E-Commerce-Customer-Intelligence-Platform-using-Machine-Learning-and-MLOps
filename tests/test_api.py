from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "E-Commerce Delivery Delay Prediction API is running"


def test_predict_endpoint():
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

    response = client.post("/predict", json=sample_order)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "delay_probability" in result
    assert "prediction_label" in result

    assert result["prediction"] in [0, 1]
    assert 0 <= result["delay_probability"] <= 1
    assert result["prediction_label"] in ["On Time", "Delayed"]