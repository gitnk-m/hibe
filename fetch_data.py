import requests
import pandas as pd

params = {
    "symbol":"BTCUSDT",
    "interval":"4h",
    "limit":5
}
res_data = requests.get("https://data-api.binance.vision/api/v3/klines", params=params)
df = pd.DataFrame([row[:6] for row in res_data.json()])
price_cols = ["open", "high", "low", "close", "volume"]
df.columns = ["open_time", "open", "high", "low", "close", "volume"]
df[price_cols] = df[price_cols].astype("float64")
df["open_time"] = pd.to_datetime(df["open_time"], utc=True, unit="ms")
print(df.dtypes)
print(df)
