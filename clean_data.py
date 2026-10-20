import pandas as pd

df = pd.read_csv("data/btc_4h.csv", parse_dates=["open_time"])
print(df.isna().sum())
print("open time duplicate: ",df["open_time"].duplicated().sum())
print("---------------------------------------")
print("is_monotonic_increasing: ",df["open_time"].is_monotonic_increasing)
print("---------------------------------------")
df.info()
print("---------------------------------------")
time_diff = df["open_time"].diff()
print(time_diff.value_counts())
target_duration = pd.Timedelta("0 days 04:00:00")
df["prev_time"] = df["open_time"].shift(1)
anomalies = df[time_diff.notna() & (time_diff != target_duration)]
print(anomalies[["prev_time","open_time"]])

print(df[((df["high"]<df["open"]) | (df["high"]<df["close"])) | ((df["low"]>df["open"]) | (df["low"]>df["close"])) | (df["volume"]<=0)])