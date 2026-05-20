## Problem Statement:

The objective of this project is to predict whether an e-commerce order will be delivered late based on customer, product, payment, freight, and order information.

Target Variable:
delivery_delay_flag

ML Type:
Supervised Machine Learning - Binary Classification

Models Used:
Logistic Regression
Random Forest

Evaluation Metrics:
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix

MLOps Practices:
Reusable pipeline
Data preprocessing pipeline
Model serialization
Train-test split
Model evaluation report

## Model Monitoring with Evidently AI

This project includes model monitoring using Evidently AI to detect data drift between reference and current datasets.

The monitoring script compares feature distributions for key order-level variables such as:

- customer_state
- product_category_name
- payment_type
- payment_value
- freight_value
- price
- product_weight_g
- product_volume
- purchase_dayofweek
- purchase_month
- purchase_year
- estimated_delivery_days

To generate the data drift report:

```bash
python monitoring/data_drift_report.py