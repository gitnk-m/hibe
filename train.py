import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

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

def evaluate(pred):
    mae = mean_absolute_error(y_test,pred)
    mse = mean_squared_error(y_test,pred)
    rmse = np.sqrt(mse)
    mae_dollar = (df["close"].iloc[split_idx:] * (y_test-pred).abs()).mean()
    direction_acc = (np.sign(pred) == np.sign(y_test)).mean()
    return mae,rmse,mae_dollar,direction_acc 

baseline_pred = np.zeros(len(y_test))
baseline_scores = evaluate(baseline_pred)
baseline_direction = (y_test > 0).mean()

model = XGBRegressor(n_estimators = 300, max_depth = 3, learning_rate = 0.05, random_state = 42)
model.fit(X_train, y_train)
model_pred = model.predict(X_test)
# print(model.feature_importances_, feature_cols)

mae, rmse, mae_dollar, direction_acc = evaluate(model_pred)

important_features = pd.Series(model.feature_importances_).sort_values(ascending=False)
print(important_features)

print(f"MAE        baseline {baseline_scores[0]:.3%}   model {mae:.3%}")
print(f"RMSE       baseline {baseline_scores[1]:.3%}   model {rmse:.3%}")
print(f"MAE_Dollar baseline {baseline_scores[2]:.3%}   model {mae_dollar:.3%}")
print(f"Direction  baseline {baseline_direction:.3%}   model {direction_acc:.3%}")