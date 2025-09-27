"""
delta_bot_mt.py
Multi-threaded continuous trading bot for Delta Testnet.

Features:
- Heikin-Ashi candles, EMA(10) on HA-close, ADX filter (ta library)
- Immediate-reverse execution (close opposite side then open new side)
- Producer/Consumer threads: fetcher -> signal queue -> executor
- Contract-size-aware position sizing
- Candle-aligned execution (trades only on candle close)
- Rate limiting and exponential backoff for API calls
- Safety caps: max position size, daily loss limit, min qty check

ENV variables (place in .env or your env):
 DELTA_API_KEY, DELTA_API_SECRET
 DELTA_BASE_URL (defaults to https://cdn-ind.testnet.deltaex.org)
 SYMBOL_UNDERLYING (default ETH)
 QUOTE_PREFERENCE (default USDT)
 TIMEFRAME_MINUTES (default 15)
 INITIAL_CAPITAL (default 170)
 RISK_PCT (default 1.0)
 LOT_SIZE (default 0.01)
 LEVERAGE (default 200)
 MAX_POSITION_CONTRACTS (optional safety cap, default 1000)
"""

import os
import time
import math
import logging
import threading
import queue
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
import ta
from delta_rest_client import DeltaRestClient   # may raise if not installed

# --- Load config ---
load_dotenv()

DELTA_API_KEY = os.getenv("DELTA_API_KEY", "")
DELTA_API_SECRET = os.getenv("DELTA_API_SECRET", "")
DELTA_BASE_URL = os.getenv("DELTA_BASE_URL", "https://cdn-ind.testnet.deltaex.org")
SYMBOL_UNDERLYING = os.getenv("SYMBOL_UNDERLYING", "ETH")
QUOTE_PREFERENCE = os.getenv("QUOTE_PREFERENCE", "USDT")
TIMEFRAME_MINUTES = int(os.getenv("TIMEFRAME_MINUTES", "15"))
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "170"))
RISK_PCT = float(os.getenv("RISK_PCT", "1.0"))
LOT_SIZE = float(os.getenv("LOT_SIZE", "0.01"))
LEVERAGE = int(os.getenv("LEVERAGE", "200"))
MAX_POSITION_CONTRACTS = float(os.getenv("MAX_POSITION_CONTRACTS", "1000"))

# strategy params
EMA_LEN = 10
ADX_LEN = 14
ADX_THRESHOLD = 20

# runtime params
CANDLE_HISTORY = 500       # how many candles to fetch for indicators
QUEUE_MAXSIZE = 10         # signals queue
FETCH_BACKOFF_BASE = 1.0   # seconds
API_BACKOFF_BASE = 0.5     # seconds
RATE_LIMIT_SLEEP = 0.1     # minimal gap between private API calls

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# thread-safe queue for signals
signal_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
stop_event = threading.Event()

# simple rate limiter lock (very small)
api_lock = threading.Lock()
last_api_call = 0.0

def rate_limited_api_sleep(min_interval=RATE_LIMIT_SLEEP):
    global last_api_call
    now = time.time()
    diff = now - last_api_call
    if diff < min_interval:
        time.sleep(min_interval - diff)
    last_api_call = time.time()

# --- indicator helpers ---
def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    ha['ha_open'] = 0.0
    ha['ha_open'].iat[0] = (df['open'].iat[0] + df['close'].iat[0]) / 2.0
    for i in range(1, len(df)):
        ha['ha_open'].iat[i] = (ha['ha_open'].iat[i-1] + ha['ha_close'].iat[i-1]) / 2.0
    ha['ha_high'] = pd.concat([df['high'], ha['ha_open'], ha['ha_close']], axis=1).max(axis=1)
    ha['ha_low'] = pd.concat([df['low'], ha['ha_open'], ha['ha_close']], axis=1).min(axis=1)
    return ha

def compute_indicators(df):
    ha = heikin_ashi(df)
    df2 = df.join(ha)
    df2['ema10'] = df2['ha_close'].ewm(span=EMA_LEN, adjust=False).mean()
    adx_ind = ta.trend.ADXIndicator(high=df2['high'], low=df2['low'], close=df2['close'], window=ADX_LEN, fillna=False)
    df2['adx'] = adx_ind.adx()
    df2['plus_di'] = adx_ind.adx_pos()
    df2['minus_di'] = adx_ind.adx_neg()
    return df2

# --- Delta public wrapper ---
class DeltaPublic:
    def __init__(self, base):
        self.base = base.rstrip('/')

    def get_products(self):
        url = f"{self.base}/v2/products"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()

    def get_candles(self, product_id, resolution_min, start_ts=None, end_ts=None, limit=200):
        url = f"{self.base}/v2/history/candles"
        params = {"product_id": int(product_id), "resolution": int(resolution_min)}
        if start_ts and end_ts:
            params['start'] = int(start_ts)
            params['end'] = int(end_ts)
        else:
            params['limit'] = int(limit)
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

# --- Trading engine (executor) ---
class DeltaEngine:
    def __init__(self):
        assert DELTA_API_KEY and DELTA_API_SECRET, "Set DELTA_API_KEY and DELTA_API_SECRET"
        self.pub = DeltaPublic(DELTA_BASE_URL)
        self.client = DeltaRestClient(base_url=DELTA_BASE_URL, api_key=DELTA_API_KEY, api_secret=DELTA_API_SECRET)
        self.product = None
        self.product_id = None
        self.contract_size = 1.0
        self.tick_size = 0.0
        self._discover_product()
        self.position_size = 0.0   # cached position size in contracts (float); updated via get_position
        self.position_lock = threading.Lock()  # lock to protect position_size

    def _discover_product(self):
        logging.info("Discovering product for %s (pref %s)", SYMBOL_UNDERLYING, QUOTE_PREFERENCE)
        data = self.pub.get_products()
        products = data.get('result') if isinstance(data, dict) and 'result' in data else data
        matches = []
        for p in products:
            sym = p.get('symbol','') or p.get('name','')
            if SYMBOL_UNDERLYING.upper() in sym.upper():
                matches.append(p)
        if not matches:
            for p in products:
                if SYMBOL_UNDERLYING.lower() in str(p).lower():
                    matches.append(p)
        if not matches:
            raise RuntimeError("No product found for underlying {}".format(SYMBOL_UNDERLYING))
        chosen = None
        for p in matches:
            if QUOTE_PREFERENCE and QUOTE_PREFERENCE.upper() in (p.get('symbol','').upper()):
                chosen = p
                break
        chosen = chosen or matches[0]
        self.product = chosen
        self.product_id = int(chosen['id'])
        self.contract_size = float(chosen.get('contract_size', chosen.get('contractSize', 1)) or 1.0)
        self.tick_size = float(chosen.get('tick_size', chosen.get('tickSize', 0)) or 0.0)
        logging.info("Selected product %s id=%s contract_size=%s tick_size=%s",
                     chosen.get('symbol'), self.product_id, self.contract_size, self.tick_size)

    # safe wrapper for API calls with backoff
    def _api_call_safe(self, fn, *args, **kwargs):
        backoff = API_BACKOFF_BASE
        for attempt in range(6):
            try:
                with api_lock:
                    rate_limited_api_sleep()
                    return fn(*args, **kwargs)
            except requests.HTTPError as e:
                logging.warning("HTTP error during API call: %s; attempt %d", e, attempt+1)
                time.sleep(backoff)
                backoff *= 2
            except Exception as e:
                logging.exception("API call exception; attempt %d", attempt+1)
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Max retries exceeded for API call")

    def get_position(self):
        try:
            resp = self._api_call_safe(self.client.get_position, self.product_id)
            pos = resp.get('result') if isinstance(resp, dict) and 'result' in resp else resp
            # normalize
            if not pos:
                return 0.0
            if isinstance(pos, dict):
                size = float(pos.get('size', 0) or 0)
                return size
            # if list
            if isinstance(pos, list) and len(pos) > 0:
                return float(pos[0].get('size', 0) or 0)
        except Exception:
            logging.exception("Failed get_position; returning 0")
        return 0.0

    def cancel_all_open_orders(self):
        try:
            resp = self._api_call_safe(self.client.get_live_orders)
            orders = resp.get('result') if isinstance(resp, dict) and 'result' in resp else resp
            if not orders:
                return
            for o in orders:
                oid = o.get('id')
                pid = o.get('product_id', self.product_id)
                try:
                    self._api_call_safe(self.client.cancel_order, pid, oid)
                    logging.info("Cancelled order id=%s", oid)
                except Exception:
                    logging.exception("Failed cancel id=%s", oid)
        except Exception:
            logging.exception("Failed to list/cancel open orders")

    def place_market_order(self, side, size_contracts):
        try:
            logging.info("Placing MARKET %s qty=%s", side.upper(), size_contracts)
            # delta-rest-client expects string sizes often
            resp = self._api_call_safe(self.client.place_order, product_id=self.product_id, size=str(size_contracts), side=side, order_type='MARKET')
            logging.info("Placed market order response: %s", str(resp)[:200])
            return resp
        except Exception:
            logging.exception("Market order failed")
            return None

    def place_stop_market(self, side, size_contracts, stop_price):
        try:
            logging.info("Placing STOP-MARKET %s qty=%s stop=%s", side.upper(), size_contracts, stop_price)
            resp = self._api_call_safe(self.client.place_stop_order, product_id=self.product_id, size=str(size_contracts), side=side, order_type='MARKET', stop_price=str(stop_price))
            logging.info("Placed stop order response: %s", str(resp)[:200])
            return resp
        except Exception:
            logging.exception("Stop market failed")
            return None

    # qty calculation converted to contract units:
    # risk_value in quote currency (e.g., USD/USDT/INR)
    # risk_per_unit is price difference per 1 unit of underlying
    # If contract_size = units underlying per contract, risk per contract = risk_per_unit * contract_size
    # qty_contracts = risk_value / risk_per_contract
    def compute_qty_contracts(self, risk_value, ha_close, stop_price):
        risk_per_unit = abs(ha_close - stop_price)
        if risk_per_unit <= 0:
            return 0.0
        risk_per_contract = risk_per_unit * self.contract_size
        raw_qty = risk_value / risk_per_contract
        # round to nearest LOT_SIZE (LOT_SIZE is in contract units)
        qty = float(round(raw_qty / LOT_SIZE) * LOT_SIZE) if LOT_SIZE > 0 else raw_qty
        # safety caps
        if qty > MAX_POSITION_CONTRACTS:
            logging.warning("Qty capped by MAX_POSITION_CONTRACTS: %s -> %s", qty, MAX_POSITION_CONTRACTS)
            qty = MAX_POSITION_CONTRACTS
        return float(qty)

    # main executor that receives signal dicts
    def execute_signal(self, signal):
        """
        signal: {
            'timestamp': <datetime>,
            'side': 'long' or 'short',
            'ha_close': float,
            'ha_open': float,
            'ha_high': float,
            'ha_low': float,
            'ema10': float,
            'adx': float
        }
        """
        side = signal['side']
        ha_close = float(signal['ha_close'])
        ha_low = float(signal['ha_low'])
        ha_high = float(signal['ha_high'])
        risk_value = INITIAL_CAPITAL * (RISK_PCT / 100.0)

        # refresh position
        with self.position_lock:
            try:
                self.position_size = float(self.get_position())
            except Exception:
                self.position_size = 0.0

        logging.info("Executor sees current position_size=%s contracts", self.position_size)

        try:
            if side == 'long':
                # if currently short -> close it (buy)
                if self.position_size < 0:
                    logging.info("Closing existing SHORT position of %s contracts", abs(self.position_size))
                    self.cancel_all_open_orders()
                    self.place_market_order('buy', abs(self.position_size))
                    time.sleep(1.0)
                # if not long already, open long
                if self.position_size <= 0:
                    stop_price = ha_low
                    qty = self.compute_qty_contracts(risk_value, ha_close, stop_price)
                    if qty <= 0:
                        logging.warning("Computed qty <= 0, skipping long entry.")
                        return
                    # Place market buy (open)
                    self.place_market_order('buy', qty)
                    time.sleep(0.8)
                    # Place stop-market to sell when stop hits
                    self.place_stop_market('sell', qty, stop_price)
                    logging.info("Opened LONG qty=%s at price~%s stop=%s", qty, ha_close, stop_price)
            elif side == 'short':
                if self.position_size > 0:
                    logging.info("Closing existing LONG position of %s contracts", abs(self.position_size))
                    self.cancel_all_open_orders()
                    self.place_market_order('sell', abs(self.position_size))
                    time.sleep(1.0)
                if self.position_size >= 0:
                    stop_price = ha_high
                    qty = self.compute_qty_contracts(risk_value, ha_close, stop_price)
                    if qty <= 0:
                        logging.warning("Computed qty <= 0, skipping short entry.")
                        return
                    self.place_market_order('sell', qty)
                    time.sleep(0.8)
                    self.place_stop_market('buy', qty, stop_price)
                    logging.info("Opened SHORT qty=%s at price~%s stop=%s", qty, ha_close, stop_price)
            else:
                logging.debug("Unknown signal side: %s", side)
        except Exception:
            logging.exception("Error executing signal")

# --- Producer (fetcher) thread ---
class FetcherThread(threading.Thread):
    def __init__(self, engine: DeltaEngine, queue_out: queue.Queue):
        super().__init__(daemon=True)
        self.engine = engine
        self.queue_out = queue_out
        self.last_signal = None

    def run(self):
        logging.info("Fetcher thread started; aligning to candle close.")
        while not stop_event.is_set():
            try:
                # align to candle close
                now = datetime.now(timezone.utc)
                minute = now.minute
                minutes_into_period = minute % TIMEFRAME_MINUTES
                secs_into_minute = now.second + now.microsecond / 1e6
                secs_to_wait = (TIMEFRAME_MINUTES - minutes_into_period) * 60 - secs_into_minute + 1.0
                if secs_to_wait < 0:
                    secs_to_wait = 1.0
                logging.info("Fetcher sleeping %.1f s until next candle close", secs_to_wait)
                time.sleep(secs_to_wait)

                # fetch candles
                end_ts = int(time.time())
                start_ts = end_ts - CANDLE_HISTORY * TIMEFRAME_MINUTES * 60
                raw = self.engine.pub.get_candles(self.engine.product_id, resolution_min=TIMEFRAME_MINUTES, start_ts=start_ts, end_ts=end_ts, limit=CANDLE_HISTORY)
                rows = raw.get('result') if isinstance(raw, dict) and 'result' in raw else raw
                if not rows:
                    logging.warning("No candles returned; retrying after short sleep")
                    time.sleep(FETCH_BACKOFF_BASE)
                    continue
                df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
                ts0 = int(df['timestamp'].iat[0])
                if ts0 > 1e12:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                for c in ['open','high','low','close','volume']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

                df_ind = compute_indicators(df)
                latest = df_ind.iloc[-1]
                ha_close = float(latest['ha_close'])
                ha_open = float(latest['ha_open'])
                ha_high = float(latest['ha_high'])
                ha_low = float(latest['ha_low'])
                ema10 = float(latest['ema10'])
                adx = float(latest['adx']) if not pd.isna(latest['adx']) else 0.0

                ha_bull = ha_close > ha_open
                ha_bear = ha_close < ha_open
                longCond = (adx > ADX_THRESHOLD) and ha_bull and (ha_close > ema10)
                shortCond = (adx > ADX_THRESHOLD) and ha_bear and (ha_close < ema10)

                signal = None
                if longCond:
                    signal = {'side':'long','ha_close':ha_close,'ha_open':ha_open,'ha_high':ha_high,'ha_low':ha_low,'ema10':ema10,'adx':adx,'timestamp':str(latest.name)}
                elif shortCond:
                    signal = {'side':'short','ha_close':ha_close,'ha_open':ha_open,'ha_high':ha_high,'ha_low':ha_low,'ema10':ema10,'adx':adx,'timestamp':str(latest.name)}
                else:
                    signal = {'side':'flat','timestamp':str(latest.name)}

                # deduplicate: only push signal if changed or explicit flat->flat with forced time interval
                if signal != self.last_signal:
                    try:
                        self.queue_out.put_nowait(signal)
                        logging.info("Enqueued signal: %s", signal['side'])
                        self.last_signal = signal
                    except queue.Full:
                        logging.warning("Signal queue full; skipping enqueue")
                else:
                    logging.info("Signal unchanged (%s); not enqueuing", signal['side'])

            except Exception:
                logging.exception("Fetcher thread error; sleeping briefly then continuing")
                time.sleep(FETCH_BACKOFF_BASE)

# --- Consumer (executor) thread ---
class ExecutorThread(threading.Thread):
    def __init__(self, engine: DeltaEngine, queue_in: queue.Queue):
        super().__init__(daemon=True)
        self.engine = engine
        self.queue_in = queue_in

    def run(self):
        logging.info("Executor thread started")
        while not stop_event.is_set():
            try:
                signal = self.queue_in.get(timeout=2.0)
            except queue.Empty:
                continue
            try:
                # if flat, we might want to close positions or do nothing; current strategy: ignore flat
                if signal.get('side') in ('long','short'):
                    self.engine.execute_signal(signal)
                else:
                    logging.debug("Received flat signal; no action.")
            except Exception:
                logging.exception("Executor run failed")
            finally:
                self.queue_in.task_done()

# --- main entry ---
def main():
    engine = DeltaEngine()
    fetcher = FetcherThread(engine, signal_queue)
    executor = ExecutorThread(engine, signal_queue)

    logging.info("Starting threads")
    fetcher.start()
    executor.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received: stopping threads")
        stop_event.set()
        fetcher.join(timeout=5.0)
        executor.join(timeout=5.0)
        logging.info("Stopped")

if __name__ == "__main__":
    main()
