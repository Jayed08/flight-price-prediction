# ✈️ Flight Price Prediction

An end-to-end machine learning project that predicts airline ticket prices using an optimized **XGBoost Regressor**. The project covers data preprocessing, hyperparameter optimization with **Optuna**, comprehensive model evaluation, explainability using **SHAP**, and deployment through an interactive **Streamlit** web application.

---

## 🚀 Live Demo

**Streamlit App:** https://flight-price-prediction-jayed.streamlit.app

---

## 📊 Project Overview

Accurately estimating flight ticket prices is an important problem for both travelers and airlines. This project develops a production-ready regression model capable of predicting flight fares based on airline, route, departure schedule, travel class, duration, stops, and booking horizon.

The workflow follows a complete machine learning pipeline:

- Data preprocessing
- Feature engineering
- Model comparison
- Hyperparameter optimization
- Cross-validation
- Model explainability
- Deployment

---

## 📁 Dataset

**Source:** Kaggle – Flight Price Prediction Dataset

- Approximately **300,000** flight records
- Domestic Indian flights
- Multiple airlines
- Economy and Business class
- Various routes and booking windows

### Features

| Feature | Description |
|----------|-------------|
| Airline | Airline operator |
| Source City | Departure city |
| Destination City | Arrival city |
| Departure Time | Time of departure |
| Arrival Time | Time of arrival |
| Stops | Number of stops |
| Class | Economy / Business |
| Duration | Flight duration |
| Days Left | Days before departure |
| Price | Target variable |

---

# ⚙️ Project Workflow

```
Data Loading
      │
      ▼
Data Cleaning
      │
      ▼
One-Hot Encoding
      │
      ▼
Train/Test Split
      │
      ▼
Baseline Linear Regression
      │
      ▼
Hyperparameter Optimization (Optuna)
      │
      ▼
Decision Tree
      │
      ▼
XGBoost
      │
      ▼
CatBoost
      │
      ▼
Model Comparison
      │
      ▼
Final Model Selection
      │
      ▼
SHAP Explainability
      │
      ▼
Streamlit Deployment
```

---

# 🤖 Models Evaluated

- Linear Regression (Baseline)
- Decision Tree Regressor
- XGBoost Regressor ⭐
- CatBoost Regressor

Hyperparameter optimization was performed using **Optuna**.

---

# 📈 Model Performance

| Model | R² | MAE | RMSE |
|------|------:|------:|------:|
| Linear Regression | 0.9115 | ₹4,575 | ₹6,753 |
| Decision Tree | 0.9833 | ₹1,290 | ₹2,936 |
| CatBoost | 0.9873 | ₹1,334 | ₹2,558 |
| **XGBoost** | **0.9889** | **₹1,133** | **₹2,395** |

### Final Test Performance (XGBoost)

| Metric | Value |
|---------|-------|
| R² Score | **0.9895** |
| MAE | **₹1,085** |
| RMSE | **₹2,330** |

---

# 🔍 Model Explainability

The final model is interpreted using:

- Native XGBoost Feature Importance
- SHAP Summary Plot
- SHAP Global Feature Importance
- Prediction vs Actual Analysis
- Residual Diagnostics

These analyses provide insight into the factors influencing predicted fares and help validate model behavior.

---

# 💻 Streamlit Application

The deployed application allows users to:

- Predict flight prices interactively
- View itinerary summary
- Explore model explainability
- Inspect SHAP visualizations
- Review residual diagnostics

---

# 🛠️ Technologies Used

### Programming

- Python

### Data Analysis

- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- CatBoost
- Optuna
- SHAP

### Deployment

- Streamlit
- Joblib

---

# 📂 Repository Structure

```
Flight-Price-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── xgb_flight_price.pkl
│   └── feature_columns.pkl
│
├── notebook/
│   └── flight_price_prediction.ipynb
│
├── assets/
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── shap_importance.png
│   ├── predicted_vs_actual.png
│   └── residual_plot.png
```

---

# ⚡ Installation

Clone the repository:

```bash
git clone https://github.com/Jayed08/flight-price-prediction.git

cd flight-price-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

**Linux/macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# 🔮 Future Improvements

- Incorporate live flight pricing APIs
- Time-series aware validation
- Model uncertainty estimation
- Automated retraining pipeline

---
