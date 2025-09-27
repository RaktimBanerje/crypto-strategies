import ccxt
import pandas as pd
import backtrader as bt
import time
from datetime import datetime

# === Step 1: Fetch ETH/USDT data from Binance ===
exchange = ccxt.binance()
symbol = "ETH/USDT"
timeframe = "3m"

since = exchange.parse8601("2025-09-01T00:00:00Z")
end_time = exchange.parse8601("2025-09-26T23:59:00Z")

limit = 10000
all_candles = []
fetch_since = since

while fetch_since < end_time:
    candles = exchange.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=limit)
    if not candles:
        break
    all_candles += candles
    print(f"Fetched {len(candles)} candles, total {len(all_candles)}")
    fetch_since = candles[-1][0] + (15 * 60 * 1000)  # next batch
    time.sleep(1)

df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("datetime", inplace=True)

# === Step 2: Strategy Class ===
class HA_EMA10_Strategy(bt.Strategy):
    params = dict(
        ema_length=10,
        risk_pct=1.0,
        lot_size=0.01,
        initial_capital=450
    )

    def __init__(self):
        # Heikin Ashi candles
        ha_close = (self.data.open + self.data.high + self.data.low + self.data.close) / 4
        self.ema10 = bt.ind.EMA(ha_close, period=self.p.ema_length)

        # Track HA candles
        self.ha_open = (self.data.open + self.data.close) / 2
        self.ha_close = ha_close

    def next(self):
        ha_bull = self.ha_close[0] > self.ha_open[0]
        ha_bear = self.ha_close[0] < self.ha_open[0]

        risk_value = self.p.initial_capital * (self.p.risk_pct / 100)

        # Long condition
        if ha_bull and self.ha_close[0] > self.ema10[0]:
            if self.position.size < 0:
                self.close()
            if self.position.size <= 0:
                stop_price = self.data.low[0]
                risk_per_unit = self.ha_close[0] - stop_price
                qty = risk_value / risk_per_unit if risk_per_unit > 0 else 0
                self.buy(size=qty)

        # Short condition
        elif ha_bear and self.ha_close[0] < self.ema10[0]:
            if self.position.size > 0:
                self.close()
            if self.position.size >= 0:
                stop_price = self.data.high[0]
                risk_per_unit = stop_price - self.ha_close[0]
                qty = risk_value / risk_per_unit if risk_per_unit > 0 else 0
                self.sell(size=qty)

# === Step 3: Backtest ===
cerebro = bt.Cerebro()
cerebro.broker.set_cash(450)

data = bt.feeds.PandasData(
    dataname=df,
    timeframe=bt.TimeFrame.Minutes,
    compression=15,
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume"
)

cerebro.adddata(data)
cerebro.addstrategy(HA_EMA10_Strategy)

print("Starting Portfolio Value:", cerebro.broker.getvalue())
cerebro.run()
print("Final Portfolio Value:", cerebro.broker.getvalue())

cerebro.plot()
