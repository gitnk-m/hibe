import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("data/btc_4h_features.csv", parse_dates=["open_time"])
feature_cols = [
    "return_1", 
    "return_lag_1", 
    "return_lag_2", 
    "return_lag_3", 
    "range_pct",
    "volume_change",
    "dist_ma_7",
    "dist_ma_21",
    "rsi_14"
    ]

X=df[feature_cols]
y=df["target"]

split_idx = int(len(df)*0.8)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

assert df["open_time"].iloc[split_idx] - df["open_time"].iloc[split_idx - 1] == pd.Timedelta("4h"), "Error in time order of train and test data"

# print(f'x_train First Open: {df["open_time"].iloc[0]}, x_train Last Open {df["open_time"].iloc[split_idx-1]}, x_test first open: {df["open_time"].iloc[split_idx]}, y_test last open: {df["open_time"].iloc[-1]}')
# print(X_train.shape, X_test.shape)
# print(y_train.shape, y_test.shape)


# Prediction Step
baseline_pred = np.zeros(len(y_test))
mae = mean_absolute_error(y_test,baseline_pred)
mse = mean_squared_error(y_test,baseline_pred)
rmse = np.sqrt(mse)
mae_dollar = (df["close"].iloc[split_idx:] * (y_test-baseline_pred).abs()).mean()
direction_acc = (y_test > 0).mean()

print(f'mae: {mae:.3%}, mse: {mse:.3%}, rmse: {rmse:.3%}, mae_doller: {mae_dollar:,.2f}')
print(f"Direction (always up): {direction_acc:.2%}")

