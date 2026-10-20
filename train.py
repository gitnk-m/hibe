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
x_train = x[:split_idx]
y_train = y[:split_idx]
x_test = x[split_idx:]
y_test = y[split_idx:]

assert df["open_time"].iloc[split_idx] < df["open_time"].iloc[split_idx+1], "Error in time order of train and test data"
print(f'x_train: {len(x_train)}, y_train: {len(y_train)}, x_test: {len(x_test)}, y_test: {len(y_test)}')
