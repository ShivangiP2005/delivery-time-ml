import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap

# 1. Load
df = pd.read_csv("../data/zomato_delivery.csv")   # adjust path if needed

# 2. Feature engineering
df['hour'] = pd.to_datetime(df['time_of_day'], format="%H:%M").dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['is_peak'] = df['hour'].apply(lambda h: 1 if (12 <= h <= 14) or (18 <= h <= 21) else 0)

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
cuisine_ohe = ohe.fit_transform(df[['cuisine']])
cuisine_cols = [f"cuisine_{c}" for c in ohe.categories_[0]]
cuisine_df = pd.DataFrame(cuisine_ohe, columns=cuisine_cols, index=df.index)
df = pd.concat([df, cuisine_df], axis=1)

features = ['distance_km','order_size','hour_sin','hour_cos','is_peak'] + cuisine_cols
X = df[features]
y = df['delivery_time']

# scale numeric for linear model
scaler = StandardScaler()
X[['distance_km','order_size']] = scaler.fit_transform(X[['distance_km','order_size']])

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model (LightGBM)
model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# metrics
def eval_summary(y_true, preds):
    print("MAE:", mean_absolute_error(y_true, preds))
    print("RMSE:", mean_squared_error(y_true, preds, squared=False))
    print("R2:", r2_score(y_true, preds))

eval_summary(y_test, pred)

# basic plot (save)
plt.figure(figsize=(6,4))
plt.scatter(y_test, pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.xlabel("Actual delivery time")
plt.ylabel("Predicted delivery time")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.savefig("../actual_vs_pred.png")
print("Plot saved: actual_vs_pred.png")

# SHAP explanation (small)
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("../shap_summary.png")
print("SHAP plot saved: shap_summary.png")
