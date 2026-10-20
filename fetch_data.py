import requests
import pandas as pd
import time
from datetime import datetime, timezone
import os

end_ms = int(datetime.now(timezone.utc).timestamp()*1000)
start_ms = end_ms - (3*365*24*60*60*1000)
all_rows = []

while start_ms < end_ms:
    params = {
        "startTime":start_ms,
        "symbol":"BTCUSDT",
        "interval":"4h",
        "limit":1000
    }
    res_data = requests.get("https://data-api.binance.vision/api/v3/klines", params=params, timeout=10)
    res_data.raise_for_status()
    rows = res_data.json()
    if not rows:
        break
    all_rows.extend(rows)

    start_ms = all_rows[-1][0] + 1
    print(f"Total fetched candles are {len(all_rows)}")

    time.sleep(0.3)

df = pd.DataFrame([row[:7] for row in all_rows])
price_cols = ["open", "high", "low", "close", "volume"]
df.columns = ["open_time"] + price_cols + ["close_time"]
df[price_cols] = df[price_cols].astype("float64")
df = df[df["close_time"]< end_ms]
df.drop(columns="close_time", inplace=True)
df["open_time"] = pd.to_datetime(df["open_time"], utc=True, unit="ms")

os.makedirs("data", exist_ok=True)
df.to_csv("data/btc_4h.csv", index=False)

print(f'Total Candles are {len(df)} and last open_time is {df["open_time"].iloc[-1]}')