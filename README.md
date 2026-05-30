# House Price Prediction using XGBoost

This project builds a machine learning model to predict house prices using structured housing data. It uses data preprocessing, feature engineering, and an XGBoost regression model to generate accurate predictions.

---

## Project Overview

The goal of this project is to predict the final sale price of houses based on various features such as:

- Property size and area
- Location-related features
- Quality and condition attributes
- Structural details

The model is trained on a labelled dataset and evaluated using regression metrics.

---

## Tech Stack

- Python 
- Pandas & NumPy (data processing)
- Matplotlib & Seaborn (visualisation)
- Scikit-learn (metrics & preprocessing tools)
- XGBoost (model training)
- Pickle (model saving)

---

## Dataset

- `train.csv` → training dataset (includes `SalePrice`)
- `test.csv` → test dataset (no target variable)

The dataset is preprocessed by:
- Handling missing values
- Encoding categorical variables
- Log-transforming the target variable to reduce skewness

---

## Data Preprocessing

- Dropped unnecessary `Id` column
- Filled missing values:
  - Numerical → median
  - Categorical → mode
- Applied log transformation:
  - `SalePrice = log1p(SalePrice)`
- One-hot encoding for categorical variables
- Combined train & test for consistent feature encoding

---

## Model

The model used is **XGBoost Regressor**, configured with:

- `n_estimators = 500`
- `learning_rate = 0.05`
- `max_depth = 5`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `random_state = 42`

XGBoost was chosen for its strong performance on structured/tabular data.

---

## Evaluation Metrics

The model is evaluated using:

- **R² Score**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

These metrics are calculated on the training data for performance estimation.

---

## Results

After training, the model is used to predict house prices on the test set. Predictions are transformed back using exponential transformation:

- `expm1()` is used to reverse the log transformation.

---

## Output Files

- `submission.csv` → final predictions for test dataset
- `house_price_model.pkl` → saved trained model

---