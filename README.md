# E-Commerce Customer Intelligence Platform using Machine Learning and MLOps

## Project Overview

This project is an end-to-end Machine Learning and MLOps implementation for predicting whether an e-commerce order is likely to be delivered late.

The project uses customer, product, payment, freight, and order-level features to predict delivery delay risk. It covers the complete ML lifecycle from data preparation and model training to API serving, Dockerization, CI validation, experiment tracking, and model monitoring.

This is designed as a practical portfolio project for Data Scientist, Machine Learning Engineer, AI Engineer, and MLOps Engineer roles.

---

## Business Problem

Late delivery can directly impact customer satisfaction, review scores, repeat purchase behavior, customer trust, and operational efficiency.

The goal of this project is to predict delivery delay risk before the actual delivery happens, so that an e-commerce business can take proactive action.

Possible business actions include:

- Prioritizing high-risk orders
- Alerting logistics teams
- Improving delivery planning
- Reducing negative customer experience
- Monitoring changes in delivery risk patterns over time

---

## Machine Learning Problem

This is a supervised binary classification problem.

### Target Variable

```text
delivery_delay_flag
```

### Target Logic

```text
1 = Delivered after estimated delivery date
0 = Delivered on time or before estimated delivery date
```

The target is created by comparing:

```text
order_delivered_customer_date > order_estimated_delivery_date
```

---

## Dataset

The project uses an e-commerce dataset based on the Olist Brazilian E-Commerce dataset.

The dataset contains information such as:

- Order details
- Customer location
- Product category
- Payment value
- Freight value
- Product price
- Product weight and size
- Order purchase timestamp
- Actual delivery date
- Estimated delivery date

The dataset file used in this project is expected at:

```text
data/Dataset by Olist.csv
```

Note: The raw dataset may be excluded from GitHub using `.gitignore` if the file is large.

---

## Implemented Scope

At the beginning, the platform was planned with four possible ML modules:

| Module | Description | Status |
|---|---|---|
| Classification | Delivery delay prediction | Implemented |
| Regression | Delivery days or freight cost prediction | Future scope |
| Clustering | Customer segmentation | Future scope |
| NLP | Review sentiment analysis | Future scope |

In the current phase, the classification module has been implemented end to end with a full MLOps workflow.

---

## Features Used

The model uses only features that are available before delivery to avoid data leakage.

```text
customer_state
product_category_name
payment_type
payment_value
freight_value
price
product_weight_g
product_volume
purchase_dayofweek
purchase_month
purchase_year
estimated_delivery_days
```

### Feature Selection Logic

Features were selected based on:

- Business relevance
- Availability before delivery
- Leakage prevention
- Interpretability
- Suitability for a clean baseline model

The following columns were not used as input features:

```text
order_delivered_customer_date
delivery_delay_days
delivery_delay_flag
order_id
customer_id
product_id
customer_unique_id
order_unique_id
```

These columns were excluded because they either directly reveal the target or may cause record memorization instead of generalizable learning.

---

## Feature Engineering

The following features were engineered from raw data:

| Feature | Purpose |
|---|---|
| `purchase_dayofweek` | Captures weekday or weekend purchase behavior |
| `purchase_month` | Captures seasonal pattern |
| `purchase_year` | Captures yearly operational trend |
| `estimated_delivery_days` | Captures promised delivery window |
| `product_volume` | Captures product size and shipping complexity |

---

## Data Leakage Prevention

The dataset is at item level, meaning one order can appear multiple times if it contains multiple products.

To avoid the same order appearing in both training and testing data, the project uses:

```text
GroupShuffleSplit
```

with:

```text
order_id
```

as the grouping key.

This prevents order-level data leakage and gives a more realistic model evaluation.

---

## Model Used

The current baseline model is:

```text
Logistic Regression
```

### Why Logistic Regression?

Logistic Regression was selected as the baseline model because it is:

- Simple
- Fast
- Interpretable
- Suitable for binary classification
- Easy to evaluate
- Easy to deploy
- Able to provide probability scores

The model uses:

```text
class_weight = balanced
```

because delayed orders are a minority class.

---

## Preprocessing Pipeline

The project uses a Scikit-learn pipeline to combine preprocessing and model training.

### Numeric Features

Numeric features are processed using:

```text
Median imputation
Standard scaling
```

### Categorical Features

Categorical features are processed using:

```text
Most frequent imputation
One-hot encoding
```

The encoder uses:

```text
handle_unknown = ignore
```

This prevents prediction failure when new unseen categories appear in API input.

---

## Model Evaluation

Since the dataset is imbalanced, accuracy alone is not used as the primary metric.

The model is evaluated using:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- Precision-recall curve
- ROC curve

### Baseline Model Performance

| Metric | Approximate Value |
|---|---:|
| ROC-AUC | 0.693 |
| PR-AUC | 0.141 |
| Precision | 0.113 |
| Recall | 0.626 |
| F1-score | 0.192 |

### Interpretation

The baseline model captures a meaningful portion of delayed orders because recall is around 62 percent. Precision is low, which means the model also creates false delay alerts.

This is acceptable for a baseline model focused on early risk detection. Future improvements can include threshold tuning, XGBoost, additional feature engineering, and hyperparameter optimization.

---

## Visualizations

The project generates model evaluation visualizations under:

```text
reports/figures/
```

Generated visualizations include:

- Target distribution
- Confusion matrix
- ROC curve
- Precision-recall curve
- Threshold tuning chart
- Model comparison charts
- Feature importance chart

These visualizations help explain class imbalance, model behavior, false positives, false negatives, and business tradeoffs.

---

## MLOps Architecture

```text
Raw Data
  ↓
Data Cleaning
  ↓
Feature Engineering
  ↓
Group-based Train-Test Split
  ↓
Scikit-learn Pipeline
  ↓
Model Training
  ↓
Model Evaluation
  ↓
MLflow Experiment Tracking
  ↓
Model Serialization using Joblib
  ↓
FastAPI Model Serving
  ↓
Docker Containerization
  ↓
Pytest Validation
  ↓
GitHub Actions CI
  ↓
Evidently AI Data Drift Monitoring
```

---

## Project Structure

```text
.
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── Dataset by Olist.csv
│
├── models/
│   └── delivery_delay_model.pkl
│
├── monitoring/
│   └── data_drift_report.py
│
├── notebooks/
│   └── 01_baseline_delay_prediction.ipynb
│
├── reports/
│   ├── figures/
│   └── monitoring/
│       └── data_drift_report.html
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── tests/
│   ├── conftest.py
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── README.md
├── mlflow.db
└── LICENSE
```

---

## Technologies Used

| Area | Tool |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| API Serving | FastAPI |
| API Server | Uvicorn |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI |
| Version Control | Git and GitHub |

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd E-Commerce-Customer-Intelligence-Platform-using-Machine-Learning-and-MLOps
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Train the Model

Run:

```bash
python src/train.py
```

This script performs:

- Data loading
- Data cleaning
- Feature engineering
- Train-test split using GroupShuffleSplit
- Model training
- Model evaluation
- MLflow logging
- Model saving

The trained model is saved at:

```text
models/delivery_delay_model.pkl
```

---

## Run Local Prediction

Run:

```bash
python src/predict.py
```

Example output:

```json
{
  "prediction": 0,
  "delay_probability": 0.3721,
  "prediction_label": "On Time"
}
```

---

## Run FastAPI Application

Start the API:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

---

## API Endpoints

### Health Check

```text
GET /
```

Response:

```json
{
  "message": "E-Commerce Delivery Delay Prediction API is running"
}
```

### Prediction Endpoint

```text
POST /predict
```

Sample input:

```json
{
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
```

Sample output:

```json
{
  "prediction": 0,
  "raw_delay_probability": 0.372066,
  "delay_probability": 0.372066,
  "delay_probability_percent": 37.2066,
  "prediction_label": "On Time"
}
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t ecommerce-delay-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 ecommerce-delay-api
```

Open:

```text
http://localhost:8000/docs
```

### Check Running Containers

```bash
docker ps
```

---

## MLflow Experiment Tracking

MLflow is used to track:

- Model parameters
- Model metrics
- Model artifact
- Training run metadata

The project uses SQLite backend:

```text
mlflow.db
```

### Start MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

Open:

```text
http://localhost:5000
```

Experiment name:

```text
delivery-delay-prediction
```

Run name:

```text
logistic_regression_baseline
```

---

## Model Monitoring with Evidently AI

This project includes data drift monitoring using Evidently AI.

The monitoring script compares reference data and current data to detect changes in feature distributions.

### Generate Data Drift Report

```bash
python monitoring/data_drift_report.py
```

The report is generated at:

```text
reports/monitoring/data_drift_report.html
```

### Why Monitoring is Important

In production, incoming data may change over time. If feature distributions shift significantly, model performance may degrade. Evidently AI helps detect such drift and supports decisions around model retraining.

---

## Automated Testing

The project uses Pytest to validate the FastAPI service.

Run tests:

```bash
pytest tests/
```

Expected result:

```text
2 passed
```

The test cases validate:

- Health endpoint response
- Prediction endpoint response
- Prediction value is valid
- Probability is between 0 and 1
- Prediction label is valid

---

## GitHub Actions CI

This project includes a GitHub Actions workflow at:

```text
.github/workflows/ci.yml
```

The CI pipeline runs automatically on push or pull request to configured branches.

It performs:

- Repository checkout
- Python setup
- Dependency installation
- Import checks
- Pytest execution
- Docker image build

A green GitHub Actions status confirms that the project works in a fresh GitHub environment.

---

## Production Deployment Approach

The project can be deployed in production using a Docker-based architecture.

Recommended deployment options:

- Render
- Railway
- AWS App Runner
- AWS ECS Fargate
- Azure Container Apps
- Google Cloud Run

A typical production deployment flow:

```text
GitHub Repository
  ↓
GitHub Actions CI/CD
  ↓
Docker Image Build
  ↓
Container Registry
  ↓
Cloud Runtime
  ↓
Public FastAPI Endpoint
```

For a portfolio deployment, Render or Railway is simple.  
For a production-style cloud deployment, AWS App Runner or ECS Fargate is recommended.

---

## Key MLOps Practices Implemented

| MLOps Area | Implementation |
|---|---|
| Reproducible Training | `src/train.py` |
| Model Serialization | Joblib |
| API Serving | FastAPI |
| Interactive API Testing | Swagger UI |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Automated Testing | Pytest |
| Continuous Integration | GitHub Actions |
| Monitoring | Evidently AI |
| Dependency Management | requirements.txt |
| Modular Structure | api, src, tests, monitoring folders |

---

## Common Commands

### Activate Environment

```bash
source .venv/bin/activate
```

### Train Model

```bash
python src/train.py
```

### Run Prediction Script

```bash
python src/predict.py
```

### Run API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
pytest tests/
```

### Build Docker Image

```bash
docker build -t ecommerce-delay-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 ecommerce-delay-api
```

### Start MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

### Generate Drift Report

```bash
python monitoring/data_drift_report.py
```

---

## Future Improvements

Planned improvements include:

- Add Random Forest and XGBoost model comparison
- Add hyperparameter tuning
- Add threshold optimization
- Add model registry workflow
- Add batch prediction pipeline
- Add customer segmentation module
- Add delivery days regression module
- Add review sentiment analysis module
- Add cloud deployment
- Add authentication for API
- Add data validation using Pandera or Great Expectations
- Add model performance monitoring when actual labels become available

---

## Interview Summary

This project demonstrates an end-to-end e-commerce delivery delay prediction system using Machine Learning and MLOps practices.

It covers:

- Business problem framing
- Data preparation
- Feature engineering
- Leakage prevention
- Model training
- Evaluation
- Experiment tracking
- Model serving
- Dockerization
- Automated testing
- CI validation
- Data drift monitoring

This project is not only a notebook-based ML model. It is structured as a production-oriented ML system.

---

## Resume Highlights

- Built an end-to-end e-commerce delivery delay prediction platform using Python, Scikit-learn, FastAPI, Docker, MLflow, Pytest, GitHub Actions, and Evidently AI.
- Developed a binary classification model to predict late deliveries using customer, product, payment, freight, and order-level features.
- Prevented data leakage using group-based train-test splitting with `order_id`.
- Implemented MLOps practices including experiment tracking, model serialization, API serving, containerization, automated testing, CI validation, and data drift monitoring.
- Deployed the trained model as a Dockerized FastAPI service with Swagger-based prediction testing.

---

## Author

Created as a Machine Learning and MLOps portfolio project.
