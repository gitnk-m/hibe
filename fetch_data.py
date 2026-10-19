import requests
import pandas as pd

payload = {
    "symbol":"BTCUSDT",
    "interval":"4h",
    "limit":5
}
res_data = requests.get("https://data-api.binance.vision/api/v3/klines", params=payload)
df = pd.DataFrame([row[:6] for row in res_data.json()])
df.columns = ["open_time", "open", "high", "low", "close", "volume"]
print(df["open"].dtype)
df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype("float64")
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
print(df)
