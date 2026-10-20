import pandas as pd

df = pd.read_csv("data/btc_4h.csv", parse_dates=["open_time"])
# Return: percentage change of close from the previous candle
df["return_1"] = df["close"].pct_change()

# Lagged return: return_lag_1, return_lag_2, return_lag_3 = return_1 shifted by 1, 2 and 3 rows.
for i in range(1,4):
    df[f'return_lag_{i}'] = df["return_1"].shift(i)

# Range: range_pct = (high − low) divided by close 
df["range_pct"] = (df["high"] - df["low"]) / df["close"]

# Volume change: volume_change = percentage change of volume.
df["volume_change"] = df["volume"].pct_change()

# Moving averages:ma_7 and ma_21 = mean of the last 7 and 21 closes.
df["ma_7"], df["ma_21"] = df["close"].rolling(7).mean(), df["close"].rolling(21).mean()

# Distance from the average: dist_ma_7 = close / ma_7 − 1, and the same for ma_21.
df["dist_ma_7"] = (df["close"]/df["ma_7"])-1
df["dist_ma_21"] = (df["close"]/df["ma_21"])-1

# the change from the previous close in dollars.
delta= df["close"].diff()

# only the positive changes, negatives set to 0
gain = delta.clip(lower=0)

# only the negative changes, turned positive, positives set to 0.
loss = -delta.clip(upper=0)

# Average both the standard way (Wilder's smoothing): 
avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

rs = avg_gain/avg_loss

df["rsi_14"] = 100-100/(1+rs)

df["target"] = df["return_1"].shift(-1)
assert df["target"].iloc[500] == df["return_1"].iloc[501], "target is not aligned with the next candle's return"
print(df[["close","return_1","target"]].tail(3))

# checking and droping rows with null values
df.dropna(inplace=True)
df = df.iloc[100:]
print(len(df))    

# saving the feature in new csv
df.to_csv("data/btc_4h_features.csv", index=False)