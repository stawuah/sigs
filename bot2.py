"""
SOL Day Trading Bot — Fixed $ Target
=====================================
Every trade targets exactly $0.50 profit or $0.25 loss.
TP and SL prices are calculated from your position size,
not from a fixed percentage — so the dollar outcome is
always the same regardless of where SOL is trading.

MATH (on $50 capital, 80% risk = $40 notional):
  SOL @ $88  ->  buy 0.4545 SOL
  TP = entry + ($0.50 / 0.4545) = entry + $1.10  (+1.25%)
  SL = entry - ($0.25 / 0.4545) = entry - $0.55  (-0.62%)

  Win  : exactly +$0.50
  Loss : exactly -$0.25
  RR   : 2:1

DAILY TARGETS:
  Goal         : +$2.00  (4 wins)
  Hard stop    : -$1.50  (6 losses)
  Max trades   : 12/day

ENTRY CONDITIONS (1m chart, all must pass):
  1. 5m EMA9 > EMA21      — only trade in direction of bigger trend
  2. 1m MACD hist cross   — histogram just flipped negative -> positive
  3. 1m RSI 42-68         — momentum present, room to run
  4. 1m volume spike      — 1.5x local 5-bar average
  5. 1m price > EMA9      — price on right side of fast MA

EXIT CONDITIONS:
  - TP      : price reaches entry + ($0.50 / qty)  -> +$0.50 exactly
  - SL      : price reaches entry - ($0.25 / qty)  -> -$0.25 exactly
  - Time SL : exit after 20 min if neither hit      -> prevents dead trades

LOOP TIMING:
  Signal scan     : every 10s
  Position check  : every 2s  (background thread)
  Cooldown        : 2 min after loss

Install:
  pip install ccxt pandas pandas-ta numpy requests python-dotenv psycopg2-binary

.env:
  EXCHANGE=binance
  API_KEY=your_key
  API_SECRET=your_secret
  TELEGRAM_TOKEN=your_token
  TELEGRAM_CHAT_ID=your_id
  DB_URL=postgresql://... (your neon connection string)
"""

import os, time, logging, requests, threading
from datetime import datetime, timezone, timedelta

import ccxt
import psycopg2
import psycopg2.extras
import numpy as np
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
EXCHANGE_ID      = os.getenv("EXCHANGE", "binance")
API_KEY          = os.getenv("API_KEY", "")
API_SECRET       = os.getenv("API_SECRET", "")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DB_URL           = os.getenv("DB_URL", "")

SYMBOL           = "SOL/USDT"
STARTING_CAPITAL = 50.0

# ── Fixed dollar targets ──────────────────────
RISK_FRACTION = 0.80     # 80% of capital = $40 notional
TARGET_WIN    = 0.50     # exactly $0.50 profit per winning trade
TARGET_LOSS   = 0.25     # exactly $0.25 loss per losing trade  (RR 2:1)

# ── Time stop ─────────────────────────────────
TIME_STOP_MINS = 20      # exit after 20 min if neither TP nor SL hit

# ── Daily limits ──────────────────────────────
MAX_TRADES_DAY      = 12
DAILY_PROFIT_TARGET = 2.00   # 4 wins -> stop, protect gains
DAILY_LOSS_LIMIT    = -1.50  # 6 losses -> halt for the day

# ── Signal filters ────────────────────────────
RSI_LOW            = 42
RSI_HIGH           = 68
VOL_MULTIPLIER     = 1.5
LOSS_COOLDOWN_MINS = 2

# ── Loop timing ───────────────────────────────
SCAN_INTERVAL     = 10
POSITION_INTERVAL = 2

# ── Sessions (informational only, never blocks trading) ───────────────────────
SESSIONS = [
    {"name": "Asian",         "start":  0, "end":  6, "stars": 3},
    {"name": "Quiet",         "start":  6, "end":  8, "stars": 1},
    {"name": "EU open",       "start":  8, "end": 12, "stars": 5},
    {"name": "EU/US overlap", "start": 12, "end": 13, "stars": 4},
    {"name": "US open",       "start": 13, "end": 17, "stars": 5},
    {"name": "US mid",        "start": 17, "end": 21, "stars": 3},
    {"name": "US close",      "start": 21, "end": 24, "stars": 2},
]
STARS_ICON = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🟢"}


# ══════════════════════════════════════════════
# DATABASE  (Neon PostgreSQL)
# ══════════════════════════════════════════════
def db_connect():
    """Return a new psycopg2 connection. Called fresh each time to avoid stale connections."""
    return psycopg2.connect(DB_URL)


def db_setup() -> None:
    """
    Create tables if they don't exist.
    Safe to call on every startup — uses IF NOT EXISTS.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS trades (
        id              SERIAL PRIMARY KEY,
        trade_date      DATE            NOT NULL,
        trade_time      TIMESTAMPTZ     NOT NULL,
        symbol          VARCHAR(20)     NOT NULL,
        entry_price     NUMERIC(18,6)   NOT NULL,
        exit_price      NUMERIC(18,6)   NOT NULL,
        qty             NUMERIC(18,6)   NOT NULL,
        notional        NUMERIC(10,2)   NOT NULL,
        pnl             NUMERIC(10,4)   NOT NULL,
        result          VARCHAR(20)     NOT NULL,
        exit_reason     VARCHAR(20)     NOT NULL,
        hold_minutes    NUMERIC(8,2)    NOT NULL,
        entry_rsi       NUMERIC(8,2),
        entry_vol_ratio NUMERIC(8,4),
        entry_macd_hist NUMERIC(12,6),
        session         VARCHAR(30),
        capital_before  NUMERIC(10,2)   NOT NULL,
        capital_after   NUMERIC(10,2)   NOT NULL,
        mode            VARCHAR(10)     NOT NULL DEFAULT 'PAPER'
    );

    CREATE TABLE IF NOT EXISTS daily_summary (
        summary_date    DATE            PRIMARY KEY,
        trades          INT             NOT NULL DEFAULT 0,
        wins            INT             NOT NULL DEFAULT 0,
        losses          INT             NOT NULL DEFAULT 0,
        win_rate        NUMERIC(6,2)    NOT NULL DEFAULT 0,
        gross_pnl       NUMERIC(10,4)   NOT NULL DEFAULT 0,
        capital_end     NUMERIC(10,2)   NOT NULL DEFAULT 0,
        best_session    VARCHAR(30)
    );
    """
    try:
        conn = db_connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
        conn.close()
        log.info("  Database tables ready.")
    except Exception as e:
        log.warning(f"  DB setup failed: {e}. Trades will NOT be saved to DB.")


def db_log_trade(
    entry_time: datetime,
    entry_price: float,
    exit_price: float,
    qty: float,
    notional: float,
    pnl: float,
    result: str,
    exit_reason: str,
    hold_minutes: float,
    capital_before: float,
    capital_after: float,
    entry_rsi: float = None,
    entry_vol_ratio: float = None,
    entry_macd_hist: float = None,
    session: str = None,
) -> None:
    """Insert one completed trade row. Silently skips on DB error — never crashes the bot."""
    mode = "LIVE" if API_KEY else "PAPER"
    try:
        conn = db_connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trades (
                        trade_date, trade_time, symbol,
                        entry_price, exit_price, qty, notional, pnl,
                        result, exit_reason, hold_minutes,
                        entry_rsi, entry_vol_ratio, entry_macd_hist,
                        session, capital_before, capital_after, mode
                    ) VALUES (
                        %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s
                    )
                """, (
                    entry_time.date(), entry_time, SYMBOL,
                    entry_price, exit_price, qty, notional, pnl,
                    result, exit_reason, round(hold_minutes, 2),
                    entry_rsi, entry_vol_ratio, entry_macd_hist,
                    session, capital_before, capital_after, mode,
                ))
        conn.close()
        log.info(f"  DB: trade logged ({result}  ${pnl:+.4f})")
    except Exception as e:
        log.warning(f"  DB write failed: {e}")


def db_upsert_daily(date, trades, wins, losses, gross_pnl, capital_end, best_session=None) -> None:
    """Upsert today's daily summary row."""
    win_rate = round(wins / trades * 100, 2) if trades > 0 else 0.0
    try:
        conn = db_connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO daily_summary
                        (summary_date, trades, wins, losses, win_rate, gross_pnl, capital_end, best_session)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (summary_date) DO UPDATE SET
                        trades       = EXCLUDED.trades,
                        wins         = EXCLUDED.wins,
                        losses       = EXCLUDED.losses,
                        win_rate     = EXCLUDED.win_rate,
                        gross_pnl    = EXCLUDED.gross_pnl,
                        capital_end  = EXCLUDED.capital_end,
                        best_session = EXCLUDED.best_session
                """, (date, trades, wins, losses, win_rate, gross_pnl, capital_end, best_session))
        conn.close()
    except Exception as e:
        log.warning(f"  DB daily summary failed: {e}")


def db_export_csv(filepath: str = "trades_export.csv") -> None:
    """
    Export all trades to a CSV file for analysis.
    Run this manually: python3 -c "from bot import db_export_csv; db_export_csv()"
    Or it runs automatically when you Ctrl+C the bot.
    """
    try:
        conn = db_connect()
        cur  = conn.cursor()
        cur.execute("""
            SELECT
                trade_date,
                trade_time AT TIME ZONE 'UTC' AS trade_time_utc,
                symbol, entry_price, exit_price, qty, notional,
                pnl, result, exit_reason, hold_minutes,
                entry_rsi, entry_vol_ratio, entry_macd_hist,
                session, capital_before, capital_after, mode
            FROM trades
            ORDER BY trade_time
        """)
        rows    = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        conn.close()

        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(filepath, index=False)
        log.info(f"  Exported {len(df)} trades to {filepath}")
        print(f"\nExported {len(df)} trades -> {filepath}")
        print("\nQuick stats:")
        if len(df):
            wins     = (df["result"] == "WIN").sum()
            total    = len(df)
            wr       = wins / total * 100
            by_week  = df.groupby(pd.to_datetime(df["trade_date"]).dt.isocalendar().week)
            print(f"  Total trades : {total}")
            print(f"  Wins         : {wins}  ({wr:.1f}%)")
            print(f"  Total P&L    : ${df['pnl'].sum():.4f}")
            print(f"  Avg hold     : {df['hold_minutes'].mean():.1f} min")
            print()
            print("  Win rate by week:")
            for week, grp in df.groupby(pd.to_datetime(df["trade_date"]).dt.isocalendar().week):
                w  = (grp["result"] == "WIN").sum()
                t  = len(grp)
                print(f"    Week {week}: {w}/{t} = {w/t*100:.0f}%")
    except Exception as e:
        log.warning(f"  CSV export failed: {e}")
        print(f"Export failed: {e}")

# ══════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scalper.log"),
    ],
)
log = logging.getLogger("DayBot")


# ══════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════
def send_telegram(msg: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=5,
        )
    except Exception as e:
        log.warning(f"Telegram error: {e}")


# ══════════════════════════════════════════════
# EXCHANGE
# ══════════════════════════════════════════════
def create_exchange() -> ccxt.Exchange:
    cls = getattr(ccxt, EXCHANGE_ID)
    ex  = cls({"apiKey": API_KEY, "secret": API_SECRET,
                "enableRateLimit": True, "options": {"defaultType": "spot"}})
    log.info(f"Connected to {EXCHANGE_ID.upper()} (spot)")
    return ex


def fetch_ohlcv(exchange, timeframe: str, limit: int = 80) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=limit)
    df  = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df


def fetch_price(exchange) -> float:
    return float(exchange.fetch_ticker(SYMBOL)["last"])


# ══════════════════════════════════════════════
# MARKET CONTEXT
# ══════════════════════════════════════════════
def get_session(utc_hour: int) -> dict:
    for s in SESSIONS:
        if s["start"] <= utc_hour < s["end"]:
            return s
    return SESSIONS[0]


def minutes_to_best_window(utc_hour: int, utc_minute: int) -> tuple[str, int]:
    best_name, best_mins = "", 9999
    for s in SESSIONS:
        if s["stars"] < 4:
            continue
        delta_h = (s["start"] - utc_hour) % 24
        if delta_h == 0 and utc_minute == 0:
            return s["name"], 0
        if delta_h == 0:
            delta_h = 24
        mins = delta_h * 60 - utc_minute
        if 0 < mins < best_mins:
            best_mins, best_name = mins, s["name"]
    return best_name, best_mins


def market_context(exchange) -> dict:
    now     = datetime.now(timezone.utc)
    session = get_session(now.hour)
    next_sess, mins_away = minutes_to_best_window(now.hour, now.minute)

    ctx = {
        "session":    session["name"],
        "stars":      session["stars"],
        "icon":       STARS_ICON.get(session["stars"], "🟡"),
        "next_sess":  next_sess,
        "mins_away":  mins_away,
        "price_chg":  0.0,
        "volatility": "unknown",
        "vol_trend":  "unknown",
        "score":      0,
        "note":       "unknown",
    }

    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        ctx["price_chg"] = round(float(ticker.get("percentage", 0) or 0), 2)

        df = fetch_ohlcv(exchange, "5m", limit=30)
        df["range_pct"] = (df["high"] - df["low"]) / df["low"] * 100

        rec_range  = float(df["range_pct"].iloc[-6:-1].mean())
        base_range = float(df["range_pct"].iloc[-30:-6].mean())
        vr  = rec_range  / base_range  if base_range  > 0 else 1.0

        rec_vol  = float(df["volume"].iloc[-6:-1].mean())
        base_vol = float(df["volume"].iloc[-30:-6].mean())
        vvr = rec_vol / base_vol if base_vol > 0 else 1.0

        ctx["volatility"] = (
            f"HIGH ({vr:.1f}x) ⚡"  if vr  > 1.5 else
            f"normal ({vr:.1f}x)"   if vr  > 0.9 else
            f"low ({vr:.1f}x) 😴"
        )
        ctx["vol_trend"] = (
            f"rising ({vvr:.1f}x) 📈" if vvr > 1.3 else
            f"steady ({vvr:.1f}x)"    if vvr > 0.7 else
            f"fading ({vvr:.1f}x) 📉"
        )

        score = 3
        if vr   > 1.5:  score += 3
        elif vr  > 1.1: score += 2
        elif vr  > 0.8: score += 1
        if vvr > 1.3:   score += 2
        elif vvr > 1.0: score += 1
        if abs(ctx["price_chg"]) > 1.5: score += 1
        score += max(0, session["stars"] - 2)
        score = min(score, 10)
        ctx["score"] = score
        ctx["note"]  = (
            f"PRIME conditions ({score}/10) 🔥" if score >= 7 else
            f"good ({score}/10) ✅"              if score >= 5 else
            f"moderate ({score}/10) 🟡"          if score >= 3 else
            f"quiet market ({score}/10) 💤"
        )
    except Exception as e:
        log.debug(f"market_context error: {e}")

    return ctx


def log_market_context(ctx: dict) -> None:
    stars_str = "★" * ctx["stars"] + "☆" * (5 - ctx["stars"])
    log.info("─" * 65)
    log.info(
        f"  MARKET  {datetime.now(timezone.utc).strftime('%H:%M UTC')}  |  "
        f"{ctx['icon']} {ctx['session']}  {stars_str}"
    )
    log.info(
        f"  Price 24h : {ctx['price_chg']:+.2f}%  |  "
        f"Volatility: {ctx['volatility']}"
    )
    log.info(
        f"  Volume    : {ctx['vol_trend']}  |  "
        f"Conditions: {ctx['note']}"
    )
    if ctx["stars"] < 4 and ctx["mins_away"] < 9999:
        wake_t = datetime.now(timezone.utc) + timedelta(minutes=ctx["mins_away"])
        log.info(
            f"  Best window: {ctx['next_sess']} in ~{ctx['mins_away']} min  "
            f"({wake_t.strftime('%H:%M UTC')})  [bot trades anytime]"
        )
    log.info("─" * 65)


# ══════════════════════════════════════════════
# SIGNAL  (1m entry + 5m trend filter)
# ══════════════════════════════════════════════
def get_signal(exchange) -> dict | None:
    try:
        # 5m trend filter
        df5      = fetch_ohlcv(exchange, "5m", limit=40)
        c5       = df5["close"].astype(float)
        ema9_5m  = float(ta.ema(c5, length=9).iloc[-2])
        ema21_5m = float(ta.ema(c5, length=21).iloc[-2])
        trend_up = ema9_5m > ema21_5m

        # 1m indicators — always iloc[-2] (last closed candle)
        df1   = fetch_ohlcv(exchange, "1m", limit=80)
        c1    = df1["close"].astype(float)
        v1    = df1["volume"].astype(float)
        price = float(c1.iloc[-2])

        rsi_val = float(ta.rsi(c1, length=14).iloc[-2])
        rsi_ok  = RSI_LOW <= rsi_val <= RSI_HIGH

        mdf        = ta.macd(c1, fast=12, slow=26, signal=9)
        hist_now   = float(mdf["MACDh_12_26_9"].iloc[-2])
        hist_prev  = float(mdf["MACDh_12_26_9"].iloc[-3])
        macd_cross = hist_prev < 0 and hist_now > 0

        local_avg = float(v1.iloc[-7:-2].mean())
        cur_vol   = float(v1.iloc[-2])
        vol_ok    = cur_vol > local_avg * VOL_MULTIPLIER

        ema9_1m  = float(ta.ema(c1, length=9).iloc[-2])
        price_ok = price > ema9_1m

        # Compact scan log
        log.info(
            f"  ${price:.4f} | "
            f"Trend={'UP' if trend_up else 'dn'} | "
            f"MACD={'CROSS' if macd_cross else f'{hist_now:.4f}'} | "
            f"RSI={rsi_val:.0f}({'ok' if rsi_ok else 'x'}) | "
            f"Vol={cur_vol/local_avg:.1f}x({'ok' if vol_ok else 'x'}) | "
            f"EMA9={'ok' if price_ok else 'x'}"
        )

        all_pass = trend_up and macd_cross and rsi_ok and vol_ok and price_ok

        if not all_pass:
            missing = []
            if not trend_up:   missing.append("5m trend down")
            if not macd_cross: missing.append(f"MACD {hist_now:.4f} (no cross)")
            if not rsi_ok:     missing.append(f"RSI {rsi_val:.0f}")
            if not vol_ok:     missing.append(f"vol {cur_vol/local_avg:.1f}x")
            if not price_ok:   missing.append("below EMA9")
            log.info(f"  Waiting: {' | '.join(missing)}")
            return None

        log.info("  *** SIGNAL CONFIRMED — entering trade ***")
        return {
            "price":     price,
            "rsi":       round(rsi_val, 1),
            "vol_ratio": round(cur_vol / local_avg, 2),
            "hist":      round(hist_now, 5),
        }

    except Exception as e:
        log.error(f"  Signal error: {e}")
        return None


# ══════════════════════════════════════════════
# ORDER EXECUTION
# ══════════════════════════════════════════════
def place_order(exchange, side: str, qty: float, price: float) -> dict | None:
    if not API_KEY:
        log.info(
            f"  [PAPER] {side.upper()} {qty:.6f} SOL @ ${price:.4f}  "
            f"(${qty * price:.2f})"
        )
        return {"id": "paper", "amount": qty}
    try:
        order = exchange.create_market_order(SYMBOL, side, qty)
        log.info(f"  Order: {order}")
        return order
    except Exception as e:
        log.error(f"  Order failed: {e}")
        send_telegram(f"Order error: {e}")
        return None


# ══════════════════════════════════════════════
# BOT STATE
# ══════════════════════════════════════════════
class BotState:
    def __init__(self):
        self.capital          = STARTING_CAPITAL
        self.daily_pnl        = 0.0
        self.trades_today     = 0
        self.wins             = 0
        self.losses           = 0
        self.day              = datetime.now(timezone.utc).date()
        self.open_position    = None
        self.cooldown_until   = None
        self._lock            = threading.Lock()
        self._last_sess_alert = ""
        self.current_session  = "unknown"  # updated by market_context

    def reset_daily(self):
        self.daily_pnl    = 0.0
        self.trades_today = 0
        self.day          = datetime.now(timezone.utc).date()
        log.info("  New day — counters reset.")
        send_telegram(f"New trading day. Capital: ${self.capital:.2f}")

    def set_cooldown(self):
        self.cooldown_until = (
            datetime.now(timezone.utc) + timedelta(minutes=LOSS_COOLDOWN_MINS)
        )

    def in_cooldown(self) -> bool:
        if not self.cooldown_until:
            return False
        remaining = (self.cooldown_until - datetime.now(timezone.utc)).total_seconds()
        if remaining > 0:
            log.info(f"  Cooldown: {remaining:.0f}s left")
            return True
        return False

    @property
    def win_rate(self) -> str:
        t = self.wins + self.losses
        return f"{self.wins / t * 100:.0f}%" if t else "—"

    def summary(self) -> str:
        return (
            f"Capital: ${self.capital:.2f} | "
            f"Day P&L: ${self.daily_pnl:+.2f} / ${DAILY_PROFIT_TARGET:.2f} goal | "
            f"Trades: {self.trades_today} | "
            f"W/L: {self.wins}/{self.losses} ({self.win_rate})"
        )


# ══════════════════════════════════════════════
# POSITION MANAGEMENT
# ══════════════════════════════════════════════
def manage_position(exchange, state: BotState, current_price: float) -> None:
    with state._lock:
        pos = state.open_position
        if not pos:
            return

        elapsed  = (datetime.now(timezone.utc) - pos["entry_time"]).total_seconds() / 60
        hit_tp   = current_price >= pos["tp"]
        hit_sl   = current_price <= pos["sl"]
        hit_time = elapsed >= TIME_STOP_MINS

        if not (hit_tp or hit_sl or hit_time):
            import time as _t
            now_ts  = _t.monotonic()
            last_p  = pos.get("_last_log_price")
            last_ts = pos.get("_last_log_ts", 0)
            moved   = last_p is None or abs(current_price - last_p) / last_p * 100 >= 0.10
            due     = (now_ts - last_ts) >= 60

            if moved or due:
                pnl_now  = pos["qty"] * (current_price - pos["entry"])
                pnl_pct  = (current_price - pos["entry"]) / pos["entry"] * 100
                dist_tp  = pos["qty"] * (pos["tp"] - current_price)
                dist_sl  = pos["qty"] * (current_price - pos["sl"])
                log.info(
                    f"  HOLDING | ${pos['entry']:.4f} -> ${current_price:.4f} "
                    f"({pnl_pct:+.2f}%) | P&L ${pnl_now:+.4f} | "
                    f"+${dist_tp:.3f} to WIN  /-${dist_sl:.3f} to LOSS | "
                    f"{elapsed:.1f}/{TIME_STOP_MINS} min"
                )
                pos["_last_log_price"] = current_price
                pos["_last_log_ts"]    = now_ts
            return

        # ── Exit ─────────────────────────────────────────────────────────
        place_order(exchange, "sell", pos["qty"], current_price)

        if hit_tp:
            pnl    = TARGET_WIN
            result = "WIN"
            emoji  = "✅"
            state.wins += 1
        elif hit_sl:
            pnl    = -TARGET_LOSS
            result = "LOSS"
            emoji  = "❌"
            state.losses += 1
            state.set_cooldown()
        else:
            # Time stop — use real price for accuracy
            pnl = round(pos["qty"] * current_price - pos["qty"] * pos["entry"], 4)
            if pnl >= 0:
                result = "TIME STOP (flat)"
                emoji  = "⏱"
                state.wins += 1
            else:
                result = "TIME STOP (loss)"
                emoji  = "⏱❌"
                state.losses += 1
                state.set_cooldown()

        state.capital      += pnl
        state.daily_pnl    += pnl
        state.trades_today += 1

        msg = (
            f"{emoji} {result}\n"
            f"Entry ${pos['entry']:.4f} -> Exit ${current_price:.4f}\n"
            f"P&L: ${pnl:+.4f} | Held: {elapsed:.1f} min\n"
            f"{state.summary()}"
        )
        log.info(msg)
        send_telegram(msg)

        # ── Log to database ───────────────────────────────────────────────
        db_log_trade(
            entry_time      = pos["entry_time"],
            entry_price     = pos["entry"],
            exit_price      = current_price,
            qty             = pos["qty"],
            notional        = round(pos["qty"] * pos["entry"], 2),
            pnl             = pnl,
            result          = result.replace("TIME STOP (flat)", "TIME_FLAT")
                                     .replace("TIME STOP (loss)", "TIME_LOSS"),
            exit_reason     = "TP" if hit_tp else ("SL" if hit_sl else "TIME"),
            hold_minutes    = elapsed,
            capital_before  = state.capital - pnl,
            capital_after   = state.capital,
            entry_rsi       = pos.get("entry_rsi"),
            entry_vol_ratio = pos.get("entry_vol_ratio"),
            entry_macd_hist = pos.get("entry_macd_hist"),
            session         = state.current_session,
        )
        db_upsert_daily(
            date         = pos["entry_time"].date(),
            trades       = state.trades_today,
            wins         = state.wins,
            losses       = state.losses,
            gross_pnl    = state.daily_pnl,
            capital_end  = state.capital,
            best_session = state.current_session,
        )
        state.open_position = None


def force_close(exchange, state: BotState, reason: str) -> None:
    with state._lock:
        pos = state.open_position
        if not pos:
            return
        try:
            price = fetch_price(exchange)
        except Exception as e:
            log.error(f"  Force close fetch failed: {e}")
            return

        place_order(exchange, "sell", pos["qty"], price)
        pnl = round(pos["qty"] * price - pos["qty"] * pos["entry"], 4)
        state.capital      += pnl
        state.daily_pnl    += pnl
        state.trades_today += 1
        if pnl >= 0:
            state.wins += 1
        else:
            state.losses += 1
            state.set_cooldown()

        msg = (
            f"⚠️ FORCE CLOSE ({reason})\n"
            f"Entry ${pos['entry']:.4f} -> Exit ${price:.4f}\n"
            f"P&L: ${pnl:+.4f}\n{state.summary()}"
        )
        log.warning(msg)
        send_telegram(msg)
        db_log_trade(
            entry_time      = pos["entry_time"],
            entry_price     = pos["entry"],
            exit_price      = price,
            qty             = pos["qty"],
            notional        = round(pos["qty"] * pos["entry"], 2),
            pnl             = pnl,
            result          = "FORCE",
            exit_reason     = "FORCE",
            hold_minutes    = (datetime.now(timezone.utc) - pos["entry_time"]).total_seconds() / 60,
            capital_before  = state.capital - pnl,
            capital_after   = state.capital,
            session         = state.current_session,
        )
        state.open_position = None


# ══════════════════════════════════════════════
# POSITION MONITOR THREAD  (every 2s)
# ══════════════════════════════════════════════
def position_monitor_loop(
    exchange_ref: list, state: BotState, stop_event: threading.Event
) -> None:
    while not stop_event.is_set():
        try:
            if state.open_position:
                p = fetch_price(exchange_ref[0])
                manage_position(exchange_ref[0], state, p)

                if state.daily_pnl <= DAILY_LOSS_LIMIT and state.open_position:
                    force_close(exchange_ref[0], state, "daily loss limit")
                    send_telegram(
                        f"🛑 Daily loss limit. P&L: ${state.daily_pnl:.2f}"
                    )
        except Exception as e:
            log.debug(f"Monitor error: {e}")
        time.sleep(POSITION_INTERVAL)


# ══════════════════════════════════════════════
# EVALUATE AND TRADE
# ══════════════════════════════════════════════
def evaluate_and_trade(exchange, state: BotState) -> None:
    today = datetime.now(timezone.utc).date()
    if today != state.day:
        state.reset_daily()

    if state.trades_today >= MAX_TRADES_DAY:
        return
    if state.daily_pnl >= DAILY_PROFIT_TARGET:
        log.info(f"  Daily goal reached (${state.daily_pnl:+.2f}). Done. 🎯")
        return
    if state.daily_pnl <= DAILY_LOSS_LIMIT:
        log.info(f"  Daily loss limit (${state.daily_pnl:+.2f}). Halted. 🛑")
        return
    if state.open_position:
        return
    if state.in_cooldown():
        return

    sig = get_signal(exchange)
    if not sig:
        return

    price    = sig["price"]
    notional = round(state.capital * RISK_FRACTION, 2)

    if notional < 5.0:
        log.warning(f"  Notional ${notional:.2f} too low — skip")
        return

    qty = round(notional / price, 6)

    # ── Fixed dollar TP and SL ────────────────────────────────────────────
    # Every trade: win = exactly +TARGET_WIN, loss = exactly -TARGET_LOSS
    tp = round(price + TARGET_WIN  / qty, 4)
    sl = round(price - TARGET_LOSS / qty, 4)

    tp_pct = (tp - price) / price * 100
    sl_pct = (price - sl) / price * 100

    msg = (
        f"⚡ TRADE  {SYMBOL}\n"
        f"{'─' * 32}\n"
        f"Entry   : ${price:.4f}\n"
        f"TP      : ${tp:.4f}  (+{tp_pct:.2f}%)  -> WIN  +${TARGET_WIN:.2f}\n"
        f"SL      : ${sl:.4f}  (-{sl_pct:.2f}%)  -> LOSS -${TARGET_LOSS:.2f}\n"
        f"Time SL : {TIME_STOP_MINS} min\n"
        f"{'─' * 32}\n"
        f"Size    : {qty:.6f} SOL  (${notional:.2f}  {RISK_FRACTION*100:.0f}% of ${state.capital:.2f})\n"
        f"RSI={sig['rsi']}  Vol={sig['vol_ratio']:.1f}x  MACD={sig['hist']:+.5f}\n"
        f"{state.summary()}"
    )
    log.info(msg)
    send_telegram(msg)

    order = place_order(exchange, "buy", qty, price)
    if order:
        with state._lock:
            state.open_position = {
                "entry":           price,
                "tp":              tp,
                "sl":              sl,
                "qty":             order.get("amount", qty),
                "entry_time":      datetime.now(timezone.utc),
                "entry_rsi":       sig["rsi"],
                "entry_vol_ratio": sig["vol_ratio"],
                "entry_macd_hist": sig["hist"],
            }


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    notional = STARTING_CAPITAL * RISK_FRACTION

    log.info("=" * 65)
    log.info("  SOL Day Trading Bot  —  Fixed $ Target")
    log.info(f"  Capital  : ${STARTING_CAPITAL:.2f}  |  "
             f"Notional : ${notional:.0f}  ({RISK_FRACTION*100:.0f}%)")
    log.info(f"  WIN  = +${TARGET_WIN:.2f} exact  |  "
             f"LOSS = -${TARGET_LOSS:.2f} exact  |  RR 2:1")
    log.info(f"  Goal : +${DAILY_PROFIT_TARGET:.2f}/day  |  "
             f"Limit: ${DAILY_LOSS_LIMIT:.2f}/day")
    log.info(f"  Time stop: {TIME_STOP_MINS} min  |  "
             f"Cooldown: {LOSS_COOLDOWN_MINS} min  |  "
             f"Max: {MAX_TRADES_DAY} trades/day")
    log.info(f"  Scan: {SCAN_INTERVAL}s  |  "
             f"Position check: {POSITION_INTERVAL}s  |  "
             f"Mode: {'PAPER' if not API_KEY else 'LIVE'}")
    log.info("=" * 65)

    exchange      = create_exchange()
    exchange_ref  = [exchange]
    state         = BotState()
    stop_event    = threading.Event()
    network_fails = 0

    ctx = market_context(exchange)
    log_market_context(ctx)

    send_telegram(
        f"SOL Day Bot live | {'PAPER' if not API_KEY else 'LIVE'}\n"
        f"Capital: ${state.capital:.2f} | Notional: ${notional:.0f}\n"
        f"Every trade: WIN +${TARGET_WIN:.2f} / LOSS -${TARGET_LOSS:.2f}\n"
        f"Goal: +${DAILY_PROFIT_TARGET:.2f}/day | Limit: ${DAILY_LOSS_LIMIT:.2f}\n"
        f"Session: {ctx['session']} | {ctx['note']}"
    )

    db_setup()

    monitor_thread = threading.Thread(
        target=position_monitor_loop,
        args=(exchange_ref, state, stop_event),
        daemon=True,
        name="PositionMonitor",
    )
    monitor_thread.start()
    log.info("  Position monitor started (2s).")

    scan_count = 0
    while True:
        try:
            scan_count += 1

            if scan_count % 18 == 1:
                ctx = market_context(exchange)
                log_market_context(ctx)
                state.current_session = ctx["session"]
                sess_key = ctx["session"]
                if ctx["stars"] >= 4 and state._last_sess_alert != sess_key:
                    state._last_sess_alert = sess_key
                    send_telegram(
                        f"{ctx['icon']} {ctx['session']} open\n"
                        f"{ctx['note']}\n"
                        f"Vol: {ctx['vol_trend']} | "
                        f"Volatility: {ctx['volatility']}"
                    )

            if not state.open_position:
                evaluate_and_trade(exchange, state)

            log.info(f"  {state.summary()}")
            network_fails = 0

        except ccxt.NetworkError as e:
            network_fails += 1
            log.warning(f"Network error #{network_fails}: {e}")
            if network_fails >= 3:
                try:
                    exchange        = create_exchange()
                    exchange_ref[0] = exchange
                    network_fails   = 0
                    send_telegram("Exchange reconnected.")
                except Exception as re:
                    log.error(f"Reconnect failed: {re}")
            time.sleep(15)
            continue

        except ccxt.ExchangeError as e:
            log.error(f"Exchange error: {e}")
            send_telegram(f"Exchange error: {e}")
            time.sleep(30)
            continue

        except KeyboardInterrupt:
            log.info("  Stopped.")
            stop_event.set()
            if state.open_position:
                force_close(exchange, state, "user stop")
            send_telegram(f"Bot stopped.\n{state.summary()}")
            db_export_csv("trades_export.csv")
            break

        except Exception as e:
            log.error(f"Unexpected error: {e}")
            time.sleep(15)
            continue

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
