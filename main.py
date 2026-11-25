"""Tiny paper-trading bot for agent-fun/bars-replay/v1 (Python 3.13, stdlib only).

Original strategy: two long-only entry modes over hourly bars, each with its own
exit discipline, plus a trailing stop and a freefall filter.

  A) Mean reversion ("buy the oversold dip"): RSI(14, Wilder) below 32 while the
     close sits under the lower Bollinger band (20 typical prices, 2 sigma), the
     last 6 hours did not fall more than 10% (no catching a knife), and the
     trailing 24h return is above -12% (the day trend is not strongly down).
  B) Momentum breakout ("ride the move"): trailing 12h return above +6% with the
     close above EMA(12) and RSI between 55 and 70 (strong, not yet exhausted).

Exits (whole position, re-checked every bar so a refused sell is retried):
  - take profit at +10% over cost, stop loss at -15% under cost;
  - trailing stop: sell once the position is worth under 88% of its peak value
    since entry;
  - RSI above 70 (overbought) or the trailing 12h return below -5% (momentum gone).
After a sell, the symbol sits out a 12-bar cooldown before it can be bought again,
so a whipsaw does not re-enter on the very next bar.

Sizing: equity split evenly over the 7 universe slots, capped by the 25% weight
cap and by the cash actually on hand. Sells go out before buys in each reply;
a sell's proceeds never fund a buy of the same reply. No orders during warm-up.
Fully deterministic: pure function of the bars, no clock, no environment, no
randomness.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

RSI_BARS = 14
BAND_BARS = 20
BAND_SIGMAS = 2.0
EMA_BARS = 12
MAX_SLIPPAGE_BPS = 100

RSI_DIP_BELOW = 32.0
RSI_OVERBOUGHT = 70.0
RSI_BREAK_MIN = 55.0
RSI_BREAK_MAX = 70.0
DIP_MAX_6H_DROP = -0.10      # skip the dip buy in a freefall
DIP_MAX_24H_DROP = -0.12     # skip the dip buy when the day trend is strongly down
BREAK_12H_MIN = 0.06         # breakout needs +6% over 12h
FADE_12H_MAX = -0.05         # exit when 12h return falls under -5%
ROI_NUM, ROI_DEN = 110, 100  # take profit +10%
STOP_NUM, STOP_DEN = 85, 100  # stop loss -15%
TRAIL_NUM, TRAIL_DEN = 88, 100  # trailing stop: 12% off the peak
COOLDOWN_BARS = 12           # bars to wait before re-entering a sold symbol


class Series:
    """Per-symbol indicator state, updated bar by bar."""

    def __init__(self):
        self.closes = []          # raw closes (ints), oldest first
        self.typical = []         # last BAND_BARS typical prices
        self.last = None
        self.first = []           # first RSI_BARS changes, while averages seed
        self.gain = None
        self.loss = None
        self.ema_warm = []
        self.ema = None

    def push(self, bar):
        c = float(bar["c"])
        self.closes.append(c)
        self.typical.append((float(bar["h"]) + float(bar["l"]) + c) / 3.0)
        if len(self.typical) > BAND_BARS:
            self.typical.pop(0)
        if self.last is not None:
            change = c - self.last
            up = change if change > 0 else 0.0
            down = -change if change < 0 else 0.0
            if self.gain is None:
                self.first.append(change)
                if len(self.first) == RSI_BARS:
                    self.gain = sum(x for x in self.first if x > 0) / RSI_BARS
                    self.loss = sum(-x for x in self.first if x < 0) / RSI_BARS
            else:
                self.gain = (self.gain * (RSI_BARS - 1) + up) / RSI_BARS
                self.loss = (self.loss * (RSI_BARS - 1) + down) / RSI_BARS
        self.last = c
        if self.ema is None:
            self.ema_warm.append(c)
            if len(self.ema_warm) == EMA_BARS:
                self.ema = sum(self.ema_warm) / EMA_BARS
        else:
            self.ema += (2.0 / (EMA_BARS + 1)) * (c - self.ema)

    def rsi(self):
        if self.gain is None:
            return None
        moved = self.gain + self.loss
        return 50.0 if moved == 0 else 100.0 * self.gain / moved

    def lower_band(self):
        if len(self.typical) < BAND_BARS:
            return None
        mean = sum(self.typical) / BAND_BARS
        var = sum((x - mean) * (x - mean) for x in self.typical) / BAND_BARS
        return mean - BAND_SIGMAS * (var ** 0.5)

    def ret(self, bars):
        """Trailing return over the last `bars` closes, or None without enough."""
        if len(self.closes) <= bars:
            return None
        then = self.closes[-1 - bars]
        return self.closes[-1] / then - 1.0 if then else None


series = {}
# Peak value (quote units) of each held position since entry, for the trailing stop.
peak = {}
# Last seq at which each symbol was sold, for the re-entry cooldown.
last_sell_seq = {}


def on_batch(state, batch):
    for bar in batch["bars"]:
        s = series.get(bar["symbol"])
        if s is None:
            s = series[bar["symbol"]] = Series()
        s.push(bar)
    if state.warming_up:
        return []

    hello = state.hello
    caps = hello["caps"]
    fees = hello["fees"]
    fresh = {b["symbol"] for b in batch["bars"]}
    max_orders = caps["max_orders_per_bar"]
    min_notional = int(fees["min_notional_quote_units"])

    sells = []
    entries = []
    for meta in hello["symbols"]:
        symbol = meta["symbol"]
        if symbol not in fresh:
            continue
        s = series[symbol]
        held = state.positions.get(symbol)
        if held is not None:
            value = state.value(symbol, held.qty)
            cost = held.cost
            if symbol not in peak or value > peak[symbol]:
                peak[symbol] = value
            r = s.rsi()
            r12 = s.ret(12)
            if (value * ROI_DEN >= cost * ROI_NUM
                    or value * STOP_DEN <= cost * STOP_NUM
                    or value * TRAIL_DEN <= peak[symbol] * TRAIL_NUM
                    or (r is not None and r > RSI_OVERBOUGHT)
                    or (r12 is not None and r12 < FADE_12H_MAX)):
                sells.append({"side": "sell", "symbol": symbol, "qty": held.qty,
                              "max_slippage_bps": MAX_SLIPPAGE_BPS})
                last_sell_seq[symbol] = state.seq
            continue
        # Not held: check the two entry modes (after the re-entry cooldown).
        if state.seq - last_sell_seq.get(symbol, -10 ** 9) < COOLDOWN_BARS:
            continue
        r = s.rsi()
        if r is None:
            continue
        lower = s.lower_band()
        r6 = s.ret(6)
        r12 = s.ret(12)
        r24 = s.ret(24)
        dip = (lower is not None and r < RSI_DIP_BELOW
               and s.last < lower
               and r6 is not None and r6 > DIP_MAX_6H_DROP
               and r24 is not None and r24 > DIP_MAX_24H_DROP)
        breakout = (r12 is not None and r12 > BREAK_12H_MIN
                    and s.ema is not None and s.last > s.ema
                    and RSI_BREAK_MIN <= r <= RSI_BREAK_MAX)
        if dip or breakout:
            entries.append(symbol)

    orders = sells[:max_orders]
    # Drop peak records for positions no longer held.
    for symbol in list(peak):
        if symbol not in state.positions:
            del peak[symbol]

    slots = min(len(hello["symbols"]), caps["max_open_positions"])
    stake = state.equity // slots
    by_weight = caps["max_weight_bps"] * state.equity // 10000
    if stake > by_weight:
        stake = by_weight
    room = slots - len(state.positions)
    cash = state.cash
    for symbol in entries:
        if room <= 0 or len(orders) >= max_orders:
            break
        size = stake if stake <= cash else cash
        if size < min_notional:
            break
        orders.append({"side": "buy", "symbol": symbol,
                       "notional_quote_units": size,
                       "max_slippage_bps": MAX_SLIPPAGE_BPS})
        cash -= size
        room -= 1
    return orders


makefun_bars.run(on_batch)
