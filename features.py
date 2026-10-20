import pandas as pd
import ta

df = pd.read_csv("data/btc_4h.csv", parse_dates=["open_time"])
# Return: percentage change of close from the previous candle
df["return_1"] = df["close"].pct_change()

# Lagged return: return_lag_1, return_lag_2, return_lag_3 = return_1 shifted by 1, 2 and 3 rows.
# df["return_lag_1"], df["return_lag_2"], df["return_lag_3"] = df["return_1"].shift(1), df["return_1"].shift(2), df["return_1"].shift(3)
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

rsi_test = ta.momentum.RSIIndicator(close = df["close"], window = 14).rsi()
diff_rsi_test = df.rsi_14 - rsi_test

# checking and droping rows with null values
print(df.isna().sum())
df.dropna(inplace=True)
print(len(df))
print(df.volume_change.describe())

# saving the feature in new csv
df.to_csv("data/btc_4h_features.csv", index=False)

print(df["rsi_14"].describe())
print(df.tail(5))





print("================================================")
print(diff_rsi_test[diff_rsi_test > 0.01].count())