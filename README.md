# Student-Performance-Factor
# 🎓 Student Exam Score Prediction

This project is a machine learning pipeline built to predict **student exam scores** based on various academic, behavioral, and socio-economic features. It evaluates and compares multiple regression models to understand which factors most strongly influence academic performance.

---

## 📌 Objectives

- Build a robust ML pipeline to predict exam scores.
- Perform feature selection to find the most influential variables.
- Compare the performance of **Linear Regression**, **Polynomial Regression**, and **XGBoost Regressor**.
- Visualize the models' evaluation and predictions.
- Provide examples of real-world student profiles and predict their scores.

---

## 📊 Dataset

The dataset includes features such as:

- `Hours_Studied`, `Attendance`, `Previous_Scores`
- Parental involvement, access to resources, tutoring sessions
- Categorical data (e.g., `School_Type`, `Gender`, `Motivation_Level`) — encoded for model compatibility

---

## 🛠️ Tools & Libraries

- **Python 3.10+**
- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`, `seaborn`
- `PolynomialFeatures`, `RFECV`, `GridSearchCV`

---

## 🚀 Model Pipeline

### 🔹 1. Data Preprocessing
- Cleaning, encoding categorical features.
- Handling missing values.
- Correlation analysis to understand feature importance.

### 🔹 2. Feature Selection
- Using `RFECV` (Recursive Feature Elimination with Cross-Validation) to find the optimal feature set for each model.

### 🔹 3. Model Training & Tuning
Three models were trained:

| Model                | Description |
|---------------------|-------------|
| **Linear Regression**   | Basic baseline model. Assumes linear relationships. |
| **Polynomial Regression** | Uses degree-2 polynomial features to capture non-linear patterns. |
| **XGBoost Regressor**    | Ensemble gradient boosting model. Tuned via GridSearchCV. |

---

## 📈 Model Evaluation

| Metric   | Linear Regression | Polynomial Regression | XGBoost |
|----------|------------------:|----------------------:|--------:|
| **MSE**  | 1.1986            | 0.1084                | 0.2091  |
| **RMSE** | 1.0948            | 0.3293                | 0.4574  |
| **MAE**  | 0.8822            | 0.2761                | 0.3641  |
| **R²**   | 0.8841            | **0.9895**            | 0.9798  |

📌 **Polynomial Regression** provided the best overall performance by capturing non-linear feature interactions.

---

## 🔍 Prediction Examples

The model was tested on hypothetical student profiles:

```plaintext
Hours_Studied | Attendance | Previous_Scores | → Predicted Exam Score
--------------|------------|-----------------|-------------------------
      3       |     70     |       60        | ~58.6 (Low performer)
      6       |     85     |       75        | ~63.2 (Average performer)
     18       |     95     |       95        | ~89.8 (High performer)
