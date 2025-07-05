# 📉 Customer Churn Prediction

This project is a machine learning application that predicts whether a customer will churn (i.e., leave a service) based on their usage patterns and account details.
🔗 The app is available at this link: https://telco-churnprediction.streamlit.app/

## 🚀 Project Overview

The project involves training a machine learning model (Decision Tree, XGBoost, or Random Forest) to predict customer churn. The model is trained using customer data and then saved in the .sav format for later use.

## 🧠 Model Features

The model was trained on the following features:

- `state`
- `account_length`
- `area_code`
- `international_plan`
- `voice_mail_plan`
- `number_vmail_messages`
- `total_day_minutes`
- `total_day_calls`
- `total_day_charge`
- `total_eve_minutes`
- `total_eve_calls`
- `total_eve_charge`
- `total_night_minutes`
- `total_night_calls`
- `total_night_charge`
- `total_intl_minutes`
- `total_intl_calls`
- `total_intl_charge`
- `number_customer_service_calls`

## 🛠 Tech Stack

- Python
- Scikit-learn / XGBoost
- Pandas / NumPy

## 📈 Performance

Accuracy: 94%

Precision: 94%

Recall: 93%
