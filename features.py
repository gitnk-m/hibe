import pandas as pd

df = pd.read_csv("data/btc_4h.csv", parse_dates=["open_time"])
#1 Return: percentage change of close from the previous candle
df["return_1"] = df["close"].pct_change()

#2 Lagged return: return_lag_1, return_lag_2, return_lag_3 = return_1 shifted by 1, 2 and 3 rows.
df["return_lag_1"], df["return_lag_2"], df["return_lag_3"] = df["return_1"].shift(1), df["return_1"].shift(2), df["return_1"].shift(3),

#3 Range: range_pct = (high − low) divided by close 
df["range_pct"] = (df["high"] - df["low"]) / df["close"]

#4 Volume change: volume_change = percentage change of volume.
df["volume_change"] = df["volume"].pct_change()

#5 Moving averages:ma_7 and ma_21 = mean of the last 7 and 21 closes.
df["ma_7"], df["ma_21"] = df["close"].rolling(7).mean(), df["close"].rolling(21).mean()

#6 Distance from the average: dist_ma_7 = close / ma_7 − 1, and the same for ma_21.
df["dist_ma_7"] = (df["close"]/df["ma_7"])-1
df["dist_ma_21"] = (df["close"]/df["ma_21"])-1

#7 checking and droping rows with null values
print(df.isna().sum())
df.dropna(inplace=True)
print(len(df))
print(df.volume_change.describe())

#8 saving the feature in new csv
df.to_csv("data/btc_4h_features.csv", index=False)
