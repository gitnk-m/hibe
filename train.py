import pandas as pd

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

x=df[feature_cols]
y=df["target"]

split_idx = int(len(df)*0.8)
X_train = x.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = x.iloc[split_idx:]
y_test = y.iloc[split_idx:]

assert df["open_time"].iloc[split_idx] - df["open_time"].iloc[split_idx - 1] == pd.Timedelta("4h"), "Error in time order of train and test data"

print(f'x_train First Open: {df["open_time"].iloc[0]}, x_train Last Open {df["close"].iloc[split_idx]}, x_test first open: {df["open_time"].iloc[0]}, y_test last open: {df["open_time"].iloc[split_idx]}')
print(f'x_test first open: {df["open_time"].iloc[0]}, y_test last open: {df["open_time"].iloc[split_idx]}')
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
