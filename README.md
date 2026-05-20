# E-Commerce Customer Intelligence Platform using Machine Learning and MLOps

## Project Overview

This project is an end-to-end Machine Learning and MLOps platform built on an e-commerce dataset. The objective is to predict whether an order is likely to be delivered late using customer, product, payment, freight, and order-level features.

The project follows a complete ML lifecycle:

- Data preparation
- Feature engineering
- Model training
- Model evaluation
- Visualization
- Experiment tracking with MLflow
- API serving using FastAPI
- Docker containerization
- Automated testing using Pytest
- CI pipeline using GitHub Actions
- Data drift monitoring using Evidently AI

---

## Business Problem

Late delivery directly affects customer satisfaction, review scores, repeat purchase behavior, and operational efficiency.

The goal of this project is to predict delivery delay risk before actual delivery, so that an e-commerce business can take early action.

Example actions:

- Prioritize high-risk orders
- Alert logistics teams
- Improve delivery planning
- Reduce negative customer experience
- Monitor operational drift over time

---

## Machine Learning Problem

This is a supervised binary classification problem.

### Target Variable

```text
delivery_delay_flag