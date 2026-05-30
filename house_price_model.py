# importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import xgboost as xgb
import pickle


# loading the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)


train.drop(["Id"], axis=1, inplace=True)
test_ids = test["Id"]
test.drop(["Id"], axis=1, inplace=True)


# missing values

# fill numeric with median
for col in train.select_dtypes(include=["int64", "float64"]).columns:
    if col != "SalePrice":
        train[col] = train[col].fillna(train[col].median())
        if col in test.columns:
            test[col] = test[col].fillna(train[col].median())

# fill categorical with mode
for col in train.select_dtypes(include=["object"]).columns:
    train[col] = train[col].fillna(train[col].mode()[0])
    if col in test.columns:
        test[col] = test[col].fillna(train[col].mode()[0])


# feature engineering: log transform the target variable to reduce skewness
train["SalePrice"] = np.log1p(train["SalePrice"])


# combine train and test for consistent encoding
X = train.drop("SalePrice", axis=1)
y = train["SalePrice"]

combined = pd.concat([X, test], axis=0)


# one hot encoding for categorical variables
combined = pd.get_dummies(combined, drop_first=True)

X = combined.iloc[:len(train), :]
test_final = combined.iloc[len(train):, :]


# model training by using XGBoost
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y)


# evaluation on training data
train_pred = model.predict(X)

r2 = r2_score(y, train_pred)
mae = mean_absolute_error(y, train_pred)
rmse = np.sqrt(mean_squared_error(y, train_pred))

print("\nMODEL PERFORMANCE")
print("R2:", r2)
print("MAE:", mae)
print("RMSE:", rmse)


# predicting test data
preds = model.predict(test_final)
preds = np.expm1(preds)  # reverse log transform


# submission file
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": preds
})

submission.to_csv("submission.csv", index=False)
print("\nSubmission file created!")


# saving the model
pickle.dump(model, open("house_price_model.pkl", "wb"))
print("Model saved!")