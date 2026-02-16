"""
Multi-strategy crypto futures engine — Python port
Provided by user (file_238). Kept as-is.

FIXES APPLIED:
  BUG-006: Removed FundingFadeStrategy stub (ID collision with new_strategies_2h.py real impl)
  BUG-007: Added bb_squeeze to BNBUSDT pair_filters allowlist
  BUG-012: Updated STRATEGIES_2H set to include all 13 strategy IDs from new_strategies_2h.py
"""

from __future__ import annotations
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

try:
    from strategy_activation_policy import detect_market_regime, should_activate_strategy
except Exception:
    detect_market_regime = None
    should_activate_strategy = None

logger = logging.getLogger(__name__)
VERBOSE_FILTER_LOG = True
_FUNDING_CTX = {}


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float = 0.0

@dataclass
class Signal:
    side: Literal["LONG", "SHORT"]
    confidence: float
    tp_percent: float
    sl_percent: float
    leverage: int
    reason: str

@dataclass
class TradeEconomics:
    total_fees: float
    net_tp: float
    net_sl: float
    tp_roi: float
    sl_roi: float
    risk_reward: float
    breakeven_move: float
    is_profitable: bool

@dataclass
class TradeSignal:
    pair: str
    strategy_id: str
    strategy_name: str
    strategy_category: str
    side: Literal["LONG", "SHORT"]
    confidence: float
    entry_price: float
    tp_price: float
    sl_price: float
    leverage: int
    trade_size: float
    reason: str
    economics: TradeEconomics
    timestamp: float = field(default_factory=time.time)
    _filtered: bool = field(default=False, repr=False)
    _filter_reason: str = field(default="", repr=False)
    _blocked_by: str = field(default="", repr=False)

@dataclass
class ActiveTrade:
    pair: str
    strategy_id: str = ""
    side: Literal["LONG", "SHORT"] = "LONG"

@dataclass
class BTCMacroStatus:
    is_macro_move: bool
    direction: Optional[Literal["LONG", "SHORT"]]
    magnitude: float
    btc_price: float = 0.0

@dataclass
class DrawdownStatus:
    paused: bool
    drawdown: float
    peak: float = 0.0
    threshold: float = 0.0
    message: Optional[str] = None

@dataclass
class ScanDiagnostics:
    raw_count: int = 0
    after_cooldown: int = 0
    after_flood: int = 0
    after_category: int = 0
    final: int = 0
    reason: str = ""

@dataclass
class ScanResult:
    signals: list[TradeSignal] = field(default_factory=list)
    filtered: list[TradeSignal] = field(default_factory=list)
    btc_macro: Optional[BTCMacroStatus] = None
    drawdown: Optional[DrawdownStatus] = None
    diagnostics: ScanDiagnostics = field(default_factory=ScanDiagnostics)

CONFIG = {
    "min_trade_size": 10,
    "min_trade_size_1m": 8,
    "max_base_margin_1m": 8,
    "max_concurrent_trades": 10,
    "max_trades_per_pair": 2,
    "risk_per_trade": 0.02,
    "strategy_max_base_usd": {
        "cmf_divergence": 12.0,
    },
    "confidence_threshold": 0.64,
    "confidence_threshold_1m": 0.60,
    "confirm_signal": True,
    "confirm_lookback_1m": 1,
    "min_volatility_pct": 0.0018,  # slightly relaxed for 1m signal flow
    "trend_ema_fast": 20,
    "trend_ema_slow": 50,
    "regime_min_trend_pct": 0.0015,
    "atr_period": 14,
    "vol_target_pct": 0.004,
    "tp_multiplier": 1.20,
    "min_tp_percent": 0.0018,
    "min_risk_reward": 1.3,
    "max_leverage": 10,
    "max_leverage_1m": 9,
    "divergence_boost": {"enabled": True, "pairs_min": 3, "boost": 0.04, "lookback": 20},
    "oi_boost": {"enabled": True, "boost": 0.06},
    "anti_signal": {"enabled": True, "threshold": 0.20, "cooldown_sec": 21600},
    "structural_quality": {"enabled": True, "vol_mult": 2.5, "sr_dist": 0.003, "boost": 0.04},
    "agreement_boost": {"enabled": True, "min_strategies": 2, "boost": 0.03},
    "adaptive_params": {
        "enabled": True,
        "update_sec": 1800,
        "regime_atr_pct": {"low": 0.0015, "high": 0.0045},
        "rsi_period": {"low": 18, "mid": 14, "high": 12},
        "rsi_oversold": {"low": 32, "mid": 30, "high": 28},
        "rsi_overbought": {"low": 68, "mid": 70, "high": 72},
        "ema_fast": {"low": 30, "mid": 20, "high": 15},
        "ema_slow": {"low": 80, "mid": 50, "high": 35}
    },
    "max_funding_long": 0.0005,
    "max_funding_short": 0.0005,
    "fees": {"maker_rate": 0.0002, "taker_rate": 0.0005},
    "correlation": {
        "max_same_side_signals": 5,
        "max_per_category": 3,
        "same_side_window_sec": 120,
        "btc_move_lookback": 5,
        "btc_move_threshold": 0.008,
        "cooldown_sec": 450,
        "post_close_cooldown_sec": 600,
        "max_drawdown_pause": 1.0,
    },
    # ── BUG-007 FIX: Added bb_squeeze to BNBUSDT allowlist ──
    "pair_filters": {
        "LINKUSDT": ["rsi_snap", "stoch_cross", "obv_divergence", "gaussian_channel"],
        "BNBUSDT": ["bb_squeeze", "rsi_snap", "vwap_bounce", "ichimoku_cloud", "keltner_reversion", "donchian_breakout", "supertrend_flip", "adx_di_cross", "fib_pullback", "cmf_divergence", "vp_poc_reversion", "pivot_bounce", "vwap_sd_reversion", "mtf_ema_ribbon", "funding_fade", "bb_kc_squeeze", "gaussian_channel"],
        "ETCUSDT": ["atr_breakout", "bb_squeeze", "bb_kc_squeeze", "gaussian_channel"],
        "EOSUSDT": ["rsi_snap", "stoch_cross", "obv_divergence", "funding_fade", "keltner_reversion", "cmf_divergence", "vp_poc_reversion", "pivot_bounce", "vwap_sd_reversion", "vwap_bounce", "engulfing_sr", "gaussian_channel"],
        "APTUSDT": ["bb_squeeze", "bb_kc_squeeze", "rsi_snap", "stoch_cross", "obv_divergence", "cmf_divergence", "keltner_reversion", "vwap_sd_reversion", "vp_poc_reversion", "pivot_bounce", "funding_fade", "gaussian_channel"],
        "FILUSDT": ["bb_squeeze", "bb_kc_squeeze", "rsi_snap", "stoch_cross", "obv_divergence", "cmf_divergence", "keltner_reversion", "vwap_sd_reversion", "vp_poc_reversion", "pivot_bounce", "funding_fade", "gaussian_channel"],
        "ICPUSDT": ["bb_squeeze", "bb_kc_squeeze", "rsi_snap", "stoch_cross", "obv_divergence", "cmf_divergence", "keltner_reversion", "vwap_sd_reversion", "vp_poc_reversion", "pivot_bounce", "funding_fade", "gaussian_channel"],
        "RUNEUSDT": ["bb_squeeze", "bb_kc_squeeze", "rsi_snap", "stoch_cross", "obv_divergence", "cmf_divergence", "keltner_reversion", "vwap_sd_reversion", "vp_poc_reversion", "pivot_bounce", "funding_fade", "gaussian_channel"],
        "GRTUSDT": ["bb_squeeze", "bb_kc_squeeze", "rsi_snap", "stoch_cross", "obv_divergence", "cmf_divergence", "keltner_reversion", "vwap_sd_reversion", "vp_poc_reversion", "pivot_bounce", "funding_fade", "gaussian_channel"],
    },
    "strategy_categories": {
        "trend": ["ema_scalp", "triple_ema", "macd_flip", "atr_breakout", "macd_money_map_trend", "ema_cci_macd_combo", "ema_ribbon_33889_pullback", "ema10_20_cci_momentum", "topdown_aoi_shift", "impulse_macd_regime_breakout", "ema50_break_pullback_continuation", "ema_rsi_stoch_tripwire", "utbot_atr_adaptive_trend", "ma_slope_crossover_sr"],
        "reversion": ["rsi_snap", "stoch_cross", "obv_divergence", "funding_fade", "gaussian_channel", "macd_money_map_reversal", "ema_cci_macd_combo", "ema10_20_cci_momentum", "impulse_macd_regime_breakout"],
        "structural": ["vwap_bounce", "engulfing_sr", "price_action_dle", "topdown_aoi_shift", "orderblock_mtf_inducement_breaker", "fvg_bos_inversion_mtf", "fib_precision_respect_met", "vsa_volume_truth"],
        "SuperStrongYtStrategies": ["ema_cci_macd_combo", "ema_ribbon_33889_pullback", "ema10_20_cci_momentum", "price_action_dle", "topdown_aoi_shift", "impulse_macd_regime_breakout", "ema50_break_pullback_continuation", "orderblock_mtf_inducement_breaker", "fvg_bos_inversion_mtf", "fib_precision_respect_met", "ema_rsi_stoch_tripwire", "utbot_atr_adaptive_trend", "ma_slope_crossover_sr", "vsa_volume_truth"],
        "trend_4h": ["weekly_vwap_trend_4h", "ichimoku_breakout_4h"],
        "reversion_4h": ["bb_rsi_reversion_4h"],
        "structural_4h": ["structure_break_ob_4h"],
    },
    "pairs": [
        "ETHUSDT", "LTCUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "BNBUSDT",
        "ADAUSDT", "MATICUSDT", "OPUSDT", "ARBUSDT", "ATOMUSDT", "BTCUSDT",
        "ETCUSDT",
        "APTUSDT", "FILUSDT", "ICPUSDT", "RUNEUSDT", "GRTUSDT",
    ],
    "timeframes": ["1m", "5m"],
    "max_1m_trades": 10,
    "max_2h_trades": 10,
    "max_4h_trades": 10,
}

# Separate config for 2-3h strategies
CONFIG_2H = {
    "confidence_threshold": 0.60,
    "confirm_signal": False,
    "min_volatility_pct": 0.005,
    "tp_multiplier": 1.10,
    "min_tp_percent": 0.005,
    "min_sl_percent": 0.003,
    "min_risk_reward": 1.5,
    "max_funding_long": 0.0005,
    "max_funding_short": 0.0005,
    "cooldown_sec": 1200,
    "post_close_cooldown_sec": 1800,
    "skip_trend_ema_filter": True,
    "skip_regime_filter": True,
    "skip_htf_alignment": True,
}

# Separate config for 4h strategies
CONFIG_4H = {
    "confidence_threshold": 0.60,
    "min_volatility_pct": 0.008,
    "min_tp_percent": 0.015,
    "min_sl_percent": 0.008,
    "min_risk_reward": 1.5,
    "max_funding_long": 0.0008,
    "max_funding_short": 0.0008,
    "cooldown_sec": 1800,
    "post_close_cooldown_sec": 3600,
    "max_leverage": 5,
}

# ── BUG-012 FIX: Full set of all 13 strategy IDs from new_strategies_2h.py ──
# (Previously missing: adx_di_cross, fib_pullback, cmf_divergence,
#  vwap_sd_reversion, mtf_ema_ribbon)
STRATEGIES_2H = {
    "ichimoku_cloud", "keltner_reversion", "donchian_breakout", "supertrend_flip",
    "vp_poc_reversion", "pivot_bounce", "funding_fade", "bb_kc_squeeze",
    "adx_di_cross", "fib_pullback", "cmf_divergence",
    "vwap_sd_reversion", "mtf_ema_ribbon",
}

def is_2h_strategy(strategy_id: str) -> bool:
    return strategy_id in STRATEGIES_2H

def is_4h_strategy(strategy_id: str) -> bool:
    for cat_key in ("trend_4h", "reversion_4h", "structural_4h"):
        if strategy_id in CONFIG.get("strategy_categories", {}).get(cat_key, []):
            return True
    return False


def cooldown_scope_for_strategy(strategy_id: str) -> str:
    if is_4h_strategy(strategy_id):
        return "4h"
    if is_2h_strategy(strategy_id):
        return "2h"
    return "1m"


def cooldown_key(pair: str, strategy_id: str = "") -> str:
    return f"{pair}|{cooldown_scope_for_strategy(strategy_id)}"


def set_funding_context(funding: dict):
    global _FUNDING_CTX
    _FUNDING_CTX = funding or {}



def clamp_leverage(lv, CONFIG):
    try:
        return min(int(lv), int(CONFIG.get("max_leverage", lv)))
    except Exception:
        return lv

def calculate_trade_economics(entry_price: float, tp_price: float, sl_price: float, side: str, trade_size: float, leverage: int) -> TradeEconomics:
    notional = trade_size * leverage
    qty = notional / entry_price
    taker = CONFIG["fees"]["taker_rate"]
    entry_fee = notional * taker
    exit_fee = notional * taker
    total_fees = entry_fee + exit_fee
    if side == "LONG":
        tp_pnl = (tp_price - entry_price) * qty
        sl_pnl = (sl_price - entry_price) * qty
    else:
        tp_pnl = (entry_price - tp_price) * qty
        sl_pnl = (entry_price - sl_price) * qty
    net_tp = tp_pnl - total_fees
    net_sl = sl_pnl - total_fees
    tp_roi = (net_tp / trade_size) * 100
    sl_roi = (net_sl / trade_size) * 100
    risk_reward = abs(net_tp) / abs(net_sl) if abs(net_sl) > 0 else 0
    breakeven_move = (total_fees / notional) * 100
    return TradeEconomics(total_fees=total_fees, net_tp=net_tp, net_sl=net_sl, tp_roi=tp_roi, sl_roi=sl_roi, risk_reward=risk_reward, breakeven_move=breakeven_move, is_profitable=net_tp > 0 and risk_reward > 1.0)

def sma(data: list[float], period: int) -> Optional[float]:
    if len(data) < period:
        return None
    return sum(data[-period:]) / period

def ema(data: list[float], period: int) -> Optional[float]:
    if len(data) < period:
        return None
    k = 2 / (period + 1)
    val = sma(data[:period], period)
    for i in range(period, len(data)):
        val = data[i] * k + val * (1 - k)
    return val

def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(len(closes) - period, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))

def macd(closes: list[float]) -> dict:
    fast = ema(closes, 12)
    slow = ema(closes, 26)
    if fast is None or slow is None:
        return {"line": 0.0, "signal": 0.0, "histogram": 0.0}
    macd_history = []
    for i in range(30, len(closes) + 1):
        f = ema(closes[:i], 12)
        s = ema(closes[:i], 26)
        if f is not None and s is not None:
            macd_history.append(f - s)
    macd_line = fast - slow
    signal_line = ema(macd_history, 9) if len(macd_history) >= 9 else 0.0
    if signal_line is None:
        signal_line = 0.0
    return {"line": macd_line, "signal": signal_line, "histogram": macd_line - signal_line}

def bollinger_bands(closes: list[float], period: int = 20, mult: float = 2.0) -> Optional[dict]:
    if len(closes) < period:
        return None
    slc = closes[-period:]
    mean = sum(slc) / period
    variance = sum((v - mean) ** 2 for v in slc) / period
    std = math.sqrt(variance)
    upper = mean + mult * std
    lower = mean - mult * std
    bandwidth = (upper - lower) / mean if mean != 0 else 0
    return {"upper": upper, "middle": mean, "lower": lower, "std": std, "bandwidth": bandwidth}

def atr(candles: list[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    total = 0.0
    for i in range(len(candles) - period, len(candles)):
        tr = max(candles[i].high - candles[i].low, abs(candles[i].high - candles[i - 1].close), abs(candles[i].low - candles[i - 1].close))
        total += tr
    return total / period

# Adaptive parameter cache (per pair)
_ADAPT_CACHE = {}
_OI_CACHE = {}

def get_adaptive_params(pair: str, candles: list[Candle]):
    cfg = CONFIG.get("adaptive_params", {})
    if not cfg.get("enabled", False):
        return None
    now = time.time()
    cache = _ADAPT_CACHE.get(pair)
    if cache and (now - cache.get("ts", 0) < cfg.get("update_sec", 7200)):
        return cache.get("params")
    price = candles[-1].close if candles else 0
    atr_val = atr(candles, cfg.get("atr_period", 14)) if candles else 0
    atr_pct = (atr_val / price) if price else 0
    low = cfg.get("regime_atr_pct", {}).get("low", 0.0015)
    high = cfg.get("regime_atr_pct", {}).get("high", 0.0045)
    if atr_pct < low:
        regime = "low"
    elif atr_pct > high:
        regime = "high"
    else:
        regime = "mid"
    params = {
        "rsi_period": cfg.get("rsi_period", {}).get(regime, 14),
        "rsi_oversold": cfg.get("rsi_oversold", {}).get(regime, 30),
        "rsi_overbought": cfg.get("rsi_overbought", {}).get(regime, 70),
        "ema_fast": cfg.get("ema_fast", {}).get(regime, 20),
        "ema_slow": cfg.get("ema_slow", {}).get(regime, 50),
    }
    _ADAPT_CACHE[pair] = {"ts": now, "params": params}
    return params


def compute_market_divergence(market_data: dict[str, list[Candle]], lookback: int = 20):
    bull = 0; bear = 0
    for pair, candles in market_data.items():
        if len(candles) < lookback + 2: continue
        closes = [c.close for c in candles]
        recent = closes[-lookback:]
        p_low = min(recent); p_prev = min(recent[:-5]) if len(recent) > 5 else min(recent)
        r_recent = rsi(closes[-(lookback+1):], 14)
        r_prev = rsi(closes[-(lookback*2):-(lookback)], 14) if len(closes) > lookback*2 else r_recent
        if p_low < p_prev and r_recent > r_prev: bull += 1
        if p_low > p_prev and r_recent < r_prev: bear += 1
    if bull > bear: return 1, bull
    if bear > bull: return -1, bear
    return 0, max(bull, bear)


def compute_oi_context(pair: str, price: float, oi: float):
    if oi is None or oi == 0: return 0
    prev = _OI_CACHE.get(pair)
    _OI_CACHE[pair] = {"price": price, "oi": oi}
    if not prev: return 0
    dp = price - prev["price"]
    doi = oi - prev["oi"]
    if dp > 0 and doi > 0: return 1  # trend confirm long
    if dp < 0 and doi < 0: return 2  # liquidation fade long
    return 0


def structural_quality_boost(candles, side, cfg):
    if len(candles) < 30: return 0.0
    closes = [c.close for c in candles]
    price = closes[-1]
    vol = candles[-1].volume
    avg_vol = sum(c.volume for c in candles[-20:]) / 20
    vol_ratio = vol / avg_vol if avg_vol else 1.0
    # simple SR: recent high/low
    recent_high = max(c.high for c in candles[-30:])
    recent_low = min(c.low for c in candles[-30:])
    dist_high = abs(recent_high - price) / price
    dist_low = abs(price - recent_low) / price
    near_sr = (dist_high < cfg.get('sr_dist', 0.003)) or (dist_low < cfg.get('sr_dist', 0.003))
    boost = 0.0
    if vol_ratio >= cfg.get('vol_mult', 2.5):
        boost += cfg.get('boost', 0.04) * 0.6
    if near_sr:
        boost += cfg.get('boost', 0.04) * 0.4
    return boost


def stochastic(candles: list[Candle], k_period: int = 14) -> dict:
    if len(candles) < k_period:
        return {"k": 50.0, "d": 50.0}
    slc = candles[-k_period:]
    high = max(c.high for c in slc)
    low = min(c.low for c in slc)
    if high == low:
        return {"k": 50.0, "d": 50.0}
    k_val = ((candles[-1].close - low) / (high - low)) * 100
    k_values = []
    for i in range(max(0, len(candles) - 3), len(candles)):
        s = candles[max(0, i - k_period + 1): i + 1]
        h = max(c.high for c in s)
        l = min(c.low for c in s)
        k_values.append(50.0 if h == l else ((candles[i].close - l) / (h - l)) * 100)
    d_val = sum(k_values) / len(k_values) if k_values else 50.0
    return {"k": k_val, "d": d_val}

def vwap(candles: list[Candle], period: int = 20) -> Optional[float]:
    slc = candles[-period:]
    cum_tp_v = 0.0
    cum_v = 0.0
    for c in slc:
        tp = (c.high + c.low + c.close) / 3
        cum_tp_v += tp * c.volume
        cum_v += c.volume
    return cum_tp_v / cum_v if cum_v > 0 else None

def obv(candles: list[Candle]) -> float:
    val = 0.0
    for i in range(1, len(candles)):
        if candles[i].close > candles[i - 1].close:
            val += candles[i].volume
        elif candles[i].close < candles[i - 1].close:
            val -= candles[i].volume
    return val

def volume_spike(candles: list[Candle], lookback: int = 20) -> float:
    if len(candles) < lookback + 1:
        return 1.0
    vols = [c.volume for c in candles[-(lookback + 1):-1]]
    avg_vol = sma(vols, lookback)
    if avg_vol is None or avg_vol == 0:
        return 1.0
    return candles[-1].volume / avg_vol

def candle_body_ratio(c: Candle) -> float:
    rng = max(1e-9, c.high - c.low)
    body = abs(c.close - c.open)
    return body / rng


def cci(candles: list[Candle], period: int = 20) -> float:
    if len(candles) < period:
        return 0.0
    tp = [(c.high + c.low + c.close) / 3.0 for c in candles]
    window = tp[-period:]
    sma_tp = sum(window) / period
    md = sum(abs(x - sma_tp) for x in window) / period
    if md == 0:
        return 0.0
    return (tp[-1] - sma_tp) / (0.015 * md)


def heikin_ashi(candles: list[Candle]) -> list[dict]:
    if not candles:
        return []
    out = []
    prev_open = (candles[0].open + candles[0].close) / 2.0
    prev_close = (candles[0].open + candles[0].high + candles[0].low + candles[0].close) / 4.0
    out.append({
        'open': prev_open,
        'close': prev_close,
        'high': max(candles[0].high, prev_open, prev_close),
        'low': min(candles[0].low, prev_open, prev_close),
    })
    for c in candles[1:]:
        ha_close = (c.open + c.high + c.low + c.close) / 4.0
        ha_open = (prev_open + prev_close) / 2.0
        ha_high = max(c.high, ha_open, ha_close)
        ha_low = min(c.low, ha_open, ha_close)
        out.append({'open': ha_open, 'close': ha_close, 'high': ha_high, 'low': ha_low})
        prev_open, prev_close = ha_open, ha_close
    return out


def gaussian_filter(series: list[float], period: int = 20) -> Optional[float]:
    """Lightweight Gaussian-like smoother via multi-pass EMA."""
    if len(series) < max(3, period):
        return None
    p1 = max(3, period // 2)
    e1 = ema(series, period)
    e2 = ema(series, p1)
    if e1 is None or e2 is None:
        return None
    # bias toward slower component for stability
    return (2 * e1 + e2) / 3


class BaseStrategy:
    id: str = ""
    name: str = ""
    timeframe: str = "1m"
    leverage: int = 10
    avg_signals_per_hour: float = 0.0
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        raise NotImplementedError

class EMAScalpStrategy(BaseStrategy):
    id = "ema_scalp"
    name = "EMA 3/8 Scalp Crossover"
    timeframe = "1m"
    leverage = 15
    avg_signals_per_hour = 1.5
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 30: return None
        closes = [c.close for c in candles]
        ema3_now = ema(closes, 3); ema8_now = ema(closes, 8)
        ema3_prev = ema(closes[:-1], 3); ema8_prev = ema(closes[:-1], 8)
        if any(v is None for v in [ema3_now, ema8_now, ema3_prev, ema8_prev]): return None
        rsi_val = rsi(closes, 7); vol_ratio = volume_spike(candles, 15)
        if ema3_prev <= ema8_prev and ema3_now > ema8_now and rsi_val < 68 and vol_ratio > 0.8:
            conf = 0.60 + min((68 - rsi_val) / 150, 0.15) + (0.05 if vol_ratio > 1.3 else 0)
            return Signal(side="LONG", confidence=conf, tp_percent=0.004, sl_percent=0.003, leverage=15, reason=f"EMA3 crossed above EMA8 | RSI(7)={rsi_val:.1f} | Vol={vol_ratio:.1f}x")
        if ema3_prev >= ema8_prev and ema3_now < ema8_now and rsi_val > 32 and vol_ratio > 0.8:
            conf = 0.60 + min((rsi_val - 32) / 150, 0.15) + (0.05 if vol_ratio > 1.3 else 0)
            return Signal(side="SHORT", confidence=conf, tp_percent=0.004, sl_percent=0.003, leverage=15, reason=f"EMA3 crossed below EMA8 | RSI(7)={rsi_val:.1f} | Vol={vol_ratio:.1f}x")
        return None

class RSISnapStrategy(BaseStrategy):
    id = "rsi_snap"; name = "RSI Snap Reversal"; timeframe = "1m"; leverage = 12; avg_signals_per_hour = 1.2
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 30: return None
        closes = [c.close for c in candles]
        params = get_adaptive_params("RSI_SNAP", candles) or {}
        rsi_p = int(params.get("rsi_period", 14))
        rsi_os = float(params.get("rsi_oversold", 30))
        rsi_ob = float(params.get("rsi_overbought", 70))
        rsi7 = rsi(closes, 7); rsiP = rsi(closes, rsi_p); vol_ratio = volume_spike(candles, 20)
        if rsi7 < (rsi_os - 10) and rsiP < rsi_os and vol_ratio > 1.3:
            conf = 0.62 + min((rsi_os - rsi7) / 80, 0.15) + min((vol_ratio - 1.3) / 5, 0.08)
            return Signal(side="LONG", confidence=conf, tp_percent=0.005, sl_percent=0.004, leverage=12, reason=f"RSI(7)={rsi7:.1f} oversold snap | RSI({rsi_p})={rsiP:.1f} | Vol={vol_ratio:.1f}x")
        if rsi7 > (rsi_ob + 10) and rsiP > rsi_ob and vol_ratio > 1.3:
            conf = 0.62 + min((rsi7 - rsi_ob) / 80, 0.15) + min((vol_ratio - 1.3) / 5, 0.08)
            return Signal(side="SHORT", confidence=conf, tp_percent=0.005, sl_percent=0.004, leverage=12, reason=f"RSI(7)={rsi7:.1f} overbought snap | RSI({rsi_p})={rsiP:.1f} | Vol={vol_ratio:.1f}x")
        return None

class BBSqueezeStrategy(BaseStrategy):
    id = "bb_squeeze"; name = "Bollinger Squeeze Breakout"; timeframe = "5m"; leverage = 10; avg_signals_per_hour = 0.5
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 40: return None
        closes = [c.close for c in candles]; price = closes[-1]
        bb = bollinger_bands(closes, 20, 2); bb_prev = bollinger_bands(closes[:-1], 20, 2)
        if bb is None or bb_prev is None: return None
        atr_val = atr(candles, 14); atr_pct = atr_val / price if price else 0
        was_tight = bb_prev["bandwidth"] < 0.012
        is_expanding = bb["bandwidth"] > bb_prev["bandwidth"] * 1.20
        # volume filter: current volume > 1.5x avg(20)
        vols = [c.volume for c in candles[-20:]] if len(candles) >= 20 else [c.volume for c in candles]
        avg_vol = sum(vols) / max(1, len(vols))
        vol_ok = candles[-1].volume > (avg_vol * 1.5) if avg_vol else False
        # trade only tight -> expanding with volume confirmation
        if not (was_tight and is_expanding and vol_ok):
            return None
        ema20 = ema(closes, 20)
        prev_close = closes[-2] if len(closes) >= 2 else price
        close_outside = (price > bb["upper"]) or (price < bb["lower"])
        pullback_long = prev_close > bb["upper"] and price >= min(bb["middle"], ema20)
        pullback_short = prev_close < bb["lower"] and price <= max(bb["middle"], ema20)
        bonus = 0.08
        if price > bb["upper"] or pullback_long:
            conf = 0.62 + bonus
            if CONFIG.get('structural_quality', {}).get('enabled', False):
                conf += structural_quality_boost(candles, 'LONG', CONFIG.get('structural_quality', {}))
            return Signal(side="LONG", confidence=conf, tp_percent=0.012, sl_percent=0.0065, leverage=clamp_leverage(10, CONFIG), reason=f"BB squeeze breakout UP | BW={bb['bandwidth'] * 100:.2f}% | ATR={atr_pct * 100:.3f}% | Vol>1.5x")
        if price < bb["lower"] or pullback_short:
            conf = 0.62 + bonus
            if CONFIG.get('structural_quality', {}).get('enabled', False):
                conf += structural_quality_boost(candles, 'SHORT', CONFIG.get('structural_quality', {}))
            return Signal(side="SHORT", confidence=conf, tp_percent=0.012, sl_percent=0.0065, leverage=clamp_leverage(10, CONFIG), reason=f"BB squeeze breakout DOWN | BW={bb['bandwidth'] * 100:.2f}% | ATR={atr_pct * 100:.3f}% | Vol>1.5x")
        return None

class MACDFlipStrategy(BaseStrategy):
    id = "macd_flip"; name = "MACD Histogram Flip"; timeframe = "1m"; leverage = 10; avg_signals_per_hour = 0.9
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 40: return None
        closes = [c.close for c in candles]; price = closes[-1]
        m = macd(closes); m_prev = macd(closes[:-1]); ema20 = ema(closes, 20)
        if ema20 is None: return None
        threshold = price * 0.0002
        if m_prev["histogram"] < 0 and m["histogram"] > threshold:
            trend_bonus = 0.06 if price > ema20 else 0
            conf = 0.57 + trend_bonus + min(abs(m["histogram"]) / (price * 0.001), 0.1)
            trend = "UP" if price > ema20 else "DOWN"
            return Signal(side="LONG", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=clamp_leverage(10, CONFIG), reason=f"MACD histogram flipped bullish | H={m['histogram']:.4f} | Trend={trend}")
        if m_prev["histogram"] > 0 and m["histogram"] < -threshold:
            trend_bonus = 0.06 if price < ema20 else 0
            conf = 0.57 + trend_bonus + min(abs(m["histogram"]) / (price * 0.001), 0.1)
            trend = "DOWN" if price < ema20 else "UP"
            return Signal(side="SHORT", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=clamp_leverage(10, CONFIG), reason=f"MACD histogram flipped bearish | H={m['histogram']:.4f} | Trend={trend}")
        return None

class MACDMoneyMapTrendStrategy(BaseStrategy):
    id = "macd_money_map_trend"; name = "MACD Money Map Trend"; timeframe = "1m"; leverage = 9; avg_signals_per_hour = 0.5
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 70: return None
        closes = [c.close for c in candles]
        price = closes[-1]
        m_now = macd(closes)
        m_prev = macd(closes[:-1])
        m_prev2 = macd(closes[:-2])
        ema50 = ema(closes, 50)
        if ema50 is None:
            return None
        # zero-line trend compass
        long_bias = m_now["line"] > 0
        short_bias = m_now["line"] < 0
        # avoid chop near zero (distance rule with dynamic floor)
        line_floor = max(price * 0.0008, abs(m_now["line"]) * 0.0)
        if abs(m_now["line"]) < line_floor:
            return None
        # crossover confirmation wait (2 bars): crossover occurred, then 2 bars kept direction
        bull_cross = (m_prev2["line"] <= m_prev2["signal"] and m_prev["line"] > m_prev["signal"] and m_now["line"] > m_now["signal"])
        bear_cross = (m_prev2["line"] >= m_prev2["signal"] and m_prev["line"] < m_prev["signal"] and m_now["line"] < m_now["signal"])
        # zero-line pullback continuation mode
        bull_pullback = (long_bias and m_prev2["line"] > 0 and m_prev["line"] < m_prev2["line"] and m_now["line"] > m_prev["line"] and m_now["line"] > 0)
        bear_pullback = (short_bias and m_prev2["line"] < 0 and m_prev["line"] > m_prev2["line"] and m_now["line"] < m_prev["line"] and m_now["line"] < 0)

        # histogram slope acceleration/deceleration
        h_slope = m_now["histogram"] - m_prev["histogram"]
        r = rsi(closes, 14)
        vol = volume_spike(candles, 20)

        # lightweight swap-zone / price structure (recent swing levels)
        recent = candles[-30:]
        swing_high = max(c.high for c in recent)
        swing_low = min(c.low for c in recent)
        near_support = abs(price - swing_low) / max(price, 1e-9) < 0.006
        near_resistance = abs(price - swing_high) / max(price, 1e-9) < 0.006

        long_ok = long_bias and (bull_cross or bull_pullback) and price > ema50 and r < 72 and vol >= 0.8
        short_ok = short_bias and (bear_cross or bear_pullback) and price < ema50 and r > 28 and vol >= 0.8

        if long_ok:
            conf = 0.64 + min(abs(m_now["histogram"]) / max(price * 0.001, 1e-9), 0.12)
            if h_slope > 0:
                conf += 0.03
            if near_support:
                conf += 0.02
            mode = "cross" if bull_cross else "pullback"
            return Signal(side="LONG", confidence=min(conf,0.90), tp_percent=0.007, sl_percent=0.0045, leverage=9, reason=f"MACD trend long ({mode}) | line>0 | hist_slope={h_slope:.5f} | RSI={r:.1f} | Vol={vol:.1f}x")
        if short_ok:
            conf = 0.64 + min(abs(m_now["histogram"]) / max(price * 0.001, 1e-9), 0.12)
            if h_slope < 0:
                conf += 0.03
            if near_resistance:
                conf += 0.02
            mode = "cross" if bear_cross else "pullback"
            return Signal(side="SHORT", confidence=min(conf,0.90), tp_percent=0.007, sl_percent=0.0045, leverage=9, reason=f"MACD trend short ({mode}) | line<0 | hist_slope={h_slope:.5f} | RSI={r:.1f} | Vol={vol:.1f}x")
        return None

class MACDMoneyMapReversalStrategy(BaseStrategy):
    id = "macd_money_map_reversal"; name = "MACD Money Map Reversal"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.4
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 80: return None
        closes = [c.close for c in candles]
        price = closes[-1]
        m_now = macd(closes)
        m_prev = macd(closes[:-1])
        r = rsi(closes, 14)
        vol = volume_spike(candles, 20)
        # Divergence windows
        recent = candles[-10:]
        prior = candles[-20:-10]
        ph_r = max(c.high for c in recent); ph_p = max(c.high for c in prior)
        pl_r = min(c.low for c in recent); pl_p = min(c.low for c in prior)
        macd_r = macd([c.close for c in recent])["line"]
        macd_p = macd([c.close for c in prior])["line"]
        # Histogram confirmation: flip or shrinking tower
        hist_flip_up = m_prev["histogram"] < 0 and m_now["histogram"] > 0
        hist_flip_dn = m_prev["histogram"] > 0 and m_now["histogram"] < 0
        hist_shrink_up = m_prev["histogram"] < 0 and m_now["histogram"] > m_prev["histogram"]
        hist_shrink_dn = m_prev["histogram"] > 0 and m_now["histogram"] < m_prev["histogram"]

        bull_div = (pl_r < pl_p) and (macd_r > macd_p)
        bear_div = (ph_r > ph_p) and (macd_r < macd_p)

        # support/resistance confluence from local structure
        swing = candles[-50:]
        sr_low = min(c.low for c in swing)
        sr_high = max(c.high for c in swing)
        near_support = abs(price - sr_low) / max(price, 1e-9) < 0.008
        near_resistance = abs(price - sr_high) / max(price, 1e-9) < 0.008

        # Bollinger confluence (200 basis as soft bonus, not hard block)
        bb200 = bollinger_bands(closes, 200, 2)
        bb_bonus_long = 0.0
        bb_bonus_short = 0.0
        if bb200 is not None:
            if price <= bb200['lower']:
                bb_bonus_long = 0.03
            if price >= bb200['upper']:
                bb_bonus_short = 0.03

        if bull_div and (hist_flip_up or hist_shrink_up) and r < 48 and vol >= 0.8:
            conf = 0.63 + min((48 - r) / 120, 0.12) + (0.03 if near_support else 0.0) + bb_bonus_long
            return Signal(side="LONG", confidence=min(conf,0.90), tp_percent=0.008, sl_percent=0.005, leverage=8, reason=f"MACD bullish divergence + hist confirm | SR={near_support} | RSI={r:.1f} | Vol={vol:.1f}x")
        if bear_div and (hist_flip_dn or hist_shrink_dn) and r > 52 and vol >= 0.8:
            conf = 0.63 + min((r - 52) / 120, 0.12) + (0.03 if near_resistance else 0.0) + bb_bonus_short
            return Signal(side="SHORT", confidence=min(conf,0.90), tp_percent=0.008, sl_percent=0.005, leverage=8, reason=f"MACD bearish divergence + hist confirm | SR={near_resistance} | RSI={r:.1f} | Vol={vol:.1f}x")
        return None

class EMACCIMACDComboStrategy(BaseStrategy):
    id = "ema_cci_macd_combo"; name = "EMA+CCI+MACD Combo"; timeframe = "1m"; leverage = 9; avg_signals_per_hour = 0.6
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 100:
            return None
        closes = [c.close for c in candles]
        price = closes[-1]

        # adaptive EMA period from recent volatility profile (bounded)
        atr_pct = (atr(candles, 14) / price) if price else 0.0
        if atr_pct < 0.002:
            ema_n = 55
        elif atr_pct < 0.006:
            ema_n = 34
        else:
            ema_n = 21

        ema_now = ema(closes, ema_n)
        if ema_now is None:
            return None

        cci_now = cci(candles, 20)
        cci_prev = cci(candles[:-1], 20)
        m_now = macd(closes)
        m_prev = macd(closes[:-1])
        hist_slope = m_now["histogram"] - m_prev["histogram"]
        vol = volume_spike(candles, 20)

        trend_up = price > ema_now and m_now["line"] > 0
        trend_dn = price < ema_now and m_now["line"] < 0

        # trend-follow continuation using CCI pullback recovery + MACD hist continuation
        long_trend = trend_up and cci_prev < 0 and cci_now > 0 and hist_slope > 0 and vol >= 0.8
        short_trend = trend_dn and cci_prev > 0 and cci_now < 0 and hist_slope < 0 and vol >= 0.8

        # reversal mode at CCI extremes with histogram flip
        long_rev = (price < ema_now and cci_now < -120 and m_prev["histogram"] < 0 and m_now["histogram"] > m_prev["histogram"] and vol >= 0.8)
        short_rev = (price > ema_now and cci_now > 120 and m_prev["histogram"] > 0 and m_now["histogram"] < m_prev["histogram"] and vol >= 0.8)

        if long_trend or long_rev:
            base = 0.62 if long_trend else 0.60
            conf = base + min(abs(cci_now) / 400, 0.10) + (0.03 if hist_slope > 0 else 0.0)
            mode = 'trend' if long_trend else 'reversal'
            return Signal(side="LONG", confidence=min(conf,0.90), tp_percent=0.007, sl_percent=0.0045, leverage=9, reason=f"EMA+CCI+MACD long ({mode}) | EMA{ema_n} | CCI={cci_now:.1f} | hist_slope={hist_slope:.5f} | Vol={vol:.1f}x")
        if short_trend or short_rev:
            base = 0.62 if short_trend else 0.60
            conf = base + min(abs(cci_now) / 400, 0.10) + (0.03 if hist_slope < 0 else 0.0)
            mode = 'trend' if short_trend else 'reversal'
            return Signal(side="SHORT", confidence=min(conf,0.90), tp_percent=0.007, sl_percent=0.0045, leverage=9, reason=f"EMA+CCI+MACD short ({mode}) | EMA{ema_n} | CCI={cci_now:.1f} | hist_slope={hist_slope:.5f} | Vol={vol:.1f}x")
        return None

class EMARibbon33889PullbackStrategy(BaseStrategy):
    id = "ema_ribbon_33889_pullback"; name = "EMA Ribbon 3/8/89 Pullback"; timeframe = "1m"; leverage = 9; avg_signals_per_hour = 0.5
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 130:
            return None
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        price = closes[-1]

        e3h = ema(highs, 3); e3l = ema(lows, 3)
        e8h = ema(highs, 8); e8l = ema(lows, 8)
        e89 = ema(closes, 89)
        if any(v is None for v in [e3h, e3l, e8h, e8l, e89]):
            return None

        # previous values for pullback re-claim/reject checks
        e3h_p = ema(highs[:-1], 3); e3l_p = ema(lows[:-1], 3)
        e8h_p = ema(highs[:-1], 8); e8l_p = ema(lows[:-1], 8)
        if any(v is None for v in [e3h_p, e3l_p, e8h_p, e8l_p]):
            return None

        vol = volume_spike(candles, 20)
        atr_pct = (atr(candles, 14) / price) if price else 0.0
        if vol < 0.8 or atr_pct < 0.0015:
            return None

        # recent trend cross (first-pullback preference): price crossed 89 within recent window
        recent = closes[-30:]
        crossed_up_recent = any(recent[i-1] <= e89 and recent[i] > e89 for i in range(1, len(recent)))
        crossed_dn_recent = any(recent[i-1] >= e89 and recent[i] < e89 for i in range(1, len(recent)))

        # pullback happened: orange dipped into/under white, now reclaimed
        pullback_long = (e3h_p <= e8h_p or e3l_p <= e8l_p) and (e3h > e8h and e3l > e8l)
        pullback_short = (e3h_p >= e8h_p or e3l_p >= e8l_p) and (e3h < e8h and e3l < e8l)

        if price > e89 and crossed_up_recent and pullback_long:
            conf = 0.64 + min((vol - 0.8) / 4, 0.08)
            return Signal(side="LONG", confidence=min(conf,0.88), tp_percent=0.009, sl_percent=0.0045, leverage=9, reason=f"EMA ribbon long reclaim | 3/8 channel above | EMA89 trend | Vol={vol:.1f}x")
        if price < e89 and crossed_dn_recent and pullback_short:
            conf = 0.64 + min((vol - 0.8) / 4, 0.08)
            return Signal(side="SHORT", confidence=min(conf,0.88), tp_percent=0.009, sl_percent=0.0045, leverage=9, reason=f"EMA ribbon short reject | 3/8 channel below | EMA89 trend | Vol={vol:.1f}x")
        return None

class EMA1020CCIMomentumStrategy(BaseStrategy):
    id = "ema10_20_cci_momentum"; name = "EMA10/20 + CCI Momentum"; timeframe = "1m"; leverage = 9; avg_signals_per_hour = 0.7
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 80:
            return None
        closes = [c.close for c in candles]
        price = closes[-1]
        e10 = ema(closes, 10)
        e20 = ema(closes, 20)
        e10_p = ema(closes[:-1], 10)
        e20_p = ema(closes[:-1], 20)
        if any(v is None for v in [e10, e20, e10_p, e20_p]):
            return None

        cci_now = cci(candles, 20)
        cci_prev = cci(candles[:-1], 20)
        vol = volume_spike(candles, 20)
        m_now = macd(closes)
        m_prev = macd(closes[:-1])
        hist_slope = m_now['histogram'] - m_prev['histogram']

        bull_cross = e10_p <= e20_p and e10 > e20
        bear_cross = e10_p >= e20_p and e10 < e20
        cci_bull_ok = cci_now > 0 or cci_prev > 0
        cci_bear_ok = cci_now < 0 or cci_prev < 0

        # swing-aware stops simulated through wider SL in volatile conditions
        atr_pct = (atr(candles, 14) / price) if price else 0.0
        sl = 0.0045 if atr_pct < 0.004 else 0.0055
        tp = max(sl * 1.8, 0.007)

        if bull_cross and cci_bull_ok and hist_slope >= 0 and vol >= 0.8:
            conf = 0.62 + min(max(cci_now, 0) / 300, 0.10) + (0.03 if hist_slope > 0 else 0)
            return Signal(side='LONG', confidence=min(conf,0.90), tp_percent=tp, sl_percent=sl, leverage=9, reason=f"EMA10/20 bull cross + CCI confirm | CCI={cci_now:.1f} | hist_slope={hist_slope:.5f} | Vol={vol:.1f}x")
        if bear_cross and cci_bear_ok and hist_slope <= 0 and vol >= 0.8:
            conf = 0.62 + min(max(-cci_now, 0) / 300, 0.10) + (0.03 if hist_slope < 0 else 0)
            return Signal(side='SHORT', confidence=min(conf,0.90), tp_percent=tp, sl_percent=sl, leverage=9, reason=f"EMA10/20 bear cross + CCI confirm | CCI={cci_now:.1f} | hist_slope={hist_slope:.5f} | Vol={vol:.1f}x")
        return None

class PriceActionDLEStrategy(BaseStrategy):
    id = "price_action_dle"; name = "Price Action DLE"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.45
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 220:
            return None
        closes = [c.close for c in candles]
        price = closes[-1]

        # Direction (HTF proxy): higher-timeframe bias via slow EMA structure
        e50 = ema(closes, 50)
        e200 = ema(closes, 200)
        if e50 is None or e200 is None:
            return None
        bullish_bias = price > e200 and e50 > e200
        bearish_bias = price < e200 and e50 < e200
        if not (bullish_bias or bearish_bias):
            return None

        # Location: premium/discount from recent swing range
        window = candles[-120:]
        swing_high = max(c.high for c in window)
        swing_low = min(c.low for c in window)
        rng = max(1e-9, swing_high - swing_low)
        eq = (swing_high + swing_low) / 2.0
        in_discount = price <= eq
        in_premium = price >= eq

        # Avoid middle-noise zone
        if abs(price - eq) / max(price, 1e-9) < 0.0015:
            return None

        # Execution (LTF): rejection candle + break close + failure continuation
        c0, c1, c2 = candles[-1], candles[-2], candles[-3]
        body = abs(c0.close - c0.open)
        wick_low = min(c0.open, c0.close) - c0.low
        wick_up = c0.high - max(c0.open, c0.close)
        rej_bull = (wick_low > body * 1.2 and c0.close > c0.open)
        rej_bear = (wick_up > body * 1.2 and c0.close < c0.open)

        break_up = c0.close > max(c1.high, c2.high)
        break_dn = c0.close < min(c1.low, c2.low)

        # failure to continue against bias: previous opposing candle weak
        prev_opp_weak_for_long = (c1.close < c1.open and candle_body_ratio(c1) < 0.45)
        prev_opp_weak_for_short = (c1.close > c1.open and candle_body_ratio(c1) < 0.45)

        vol = volume_spike(candles, 20)
        if vol < 0.8:
            return None

        # 3R profile using dynamic SL from ATR
        atr_pct = (atr(candles, 14) / price) if price else 0.0
        sl = max(0.0035, min(0.0075, atr_pct * 0.9))
        tp = sl * 3.0

        if bullish_bias and in_discount and rej_bull and break_up and prev_opp_weak_for_long:
            conf = 0.64 + (0.04 if vol > 1.2 else 0.0)
            return Signal(side='LONG', confidence=min(conf,0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"DLE long | discount+rejection+break | eq={eq:.5f} | Vol={vol:.1f}x")
        if bearish_bias and in_premium and rej_bear and break_dn and prev_opp_weak_for_short:
            conf = 0.64 + (0.04 if vol > 1.2 else 0.0)
            return Signal(side='SHORT', confidence=min(conf,0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"DLE short | premium+rejection+break | eq={eq:.5f} | Vol={vol:.1f}x")
        return None


# ── BUG-006 FIX: FundingFadeStrategy stub REMOVED ──
# The real FundingRateFadeStrategy lives in new_strategies_2h.py with id="funding_fade".
# The old stub here (which always returned None) collided with it, preventing the
# real implementation from loading via load_2h_strategies() because of the
# `if s.id not in existing` guard. Stub class deleted entirely.


class LiquidationCascadeStrategy(BaseStrategy):
    id = "liquidation_cascade"; name = "Liquidation Cascade (Proxy)"; timeframe = "1m"; leverage = 10; avg_signals_per_hour = 0.4
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 30: return None
        closes = [c.close for c in candles]; price = closes[-1]
        atr_val = atr(candles, 14); atr_pct = atr_val / price if price else 0
        last = candles[-1]
        rng = (last.high - last.low) if last else 0
        vol_ratio = volume_spike(candles, 20)
        if atr_val == 0: return None
        impulse = rng / atr_val
        # strong impulse + volume spike = liquidation proxy
        if impulse > 2.0 and vol_ratio > 2.0:
            if last.close < last.open and (last.close - last.low) / max(1e-9, rng) < 0.3:
                conf = 0.60 + min((impulse - 2.0) * 0.05, 0.10)
                return Signal(side="SHORT", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=clamp_leverage(10, CONFIG), reason=f"Liquidation cascade proxy DOWN | Impulse={impulse:.2f}x | Vol={vol_ratio:.1f}x")
            if last.close > last.open and (last.high - last.close) / max(1e-9, rng) < 0.3:
                conf = 0.60 + min((impulse - 2.0) * 0.05, 0.10)
                return Signal(side="LONG", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=clamp_leverage(10, CONFIG), reason=f"Liquidation cascade proxy UP | Impulse={impulse:.2f}x | Vol={vol_ratio:.1f}x")
        return None
class VWAPBounceStrategy(BaseStrategy):
    id = "vwap_bounce"; name = "VWAP Bounce Scalp"; timeframe = "1m"; leverage = 10; avg_signals_per_hour = 0.8
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 40: return None
        closes = [c.close for c in candles]; price = closes[-1]
        vwap_val = vwap(candles, 30)
        vwap_prev = vwap(candles[:-1], 30)
        if vwap_val is None or vwap_prev is None: return None
        dist = (price - vwap_val) / vwap_val; rsi_val = rsi(closes, 14); prev = candles[-2]
        # filters: not too close, some volatility, volume confirmation
        min_dist = 0.006; max_dist = 0.02
        atr_val = atr(candles, 14); atr_pct = atr_val / price if price else 0
        vols = [c.volume for c in candles[-20:]] if len(candles) >= 20 else [c.volume for c in candles]
        avg_vol = sum(vols) / max(1, len(vols))
        vol_ok = candles[-1].volume > (avg_vol * 1.5) if avg_vol else False
        if not (min_dist <= abs(dist) <= max_dist):
            return None
        if atr_pct < 0.0025:
            return None
        if not vol_ok:
            return None
        vwap_rising = vwap_val > vwap_prev
        vwap_falling = vwap_val < vwap_prev
        if prev.low <= vwap_val * 1.001 and price > vwap_val and 42 < rsi_val < 65 and vwap_rising:
            conf = 0.60 + min((65 - rsi_val) / 200, 0.08)
            if CONFIG.get('structural_quality', {}).get('enabled', False):
                conf += structural_quality_boost(candles, 'LONG', CONFIG.get('structural_quality', {}))
            return Signal(side="LONG", confidence=conf, tp_percent=0.005, sl_percent=0.003, leverage=clamp_leverage(10, CONFIG), reason=f"VWAP bounce long | Dist={dist * 100:.3f}% | RSI={rsi_val:.1f} | ATR%={atr_pct*100:.3f}% | Vol>1.5x")
        if prev.high >= vwap_val * 0.999 and price < vwap_val and 35 < rsi_val < 58 and vwap_falling:
            conf = 0.60 + min((rsi_val - 35) / 200, 0.08)
            if CONFIG.get('structural_quality', {}).get('enabled', False):
                conf += structural_quality_boost(candles, 'SHORT', CONFIG.get('structural_quality', {}))
            return Signal(side="SHORT", confidence=conf, tp_percent=0.005, sl_percent=0.003, leverage=clamp_leverage(10, CONFIG), reason=f"VWAP rejection short | Dist={dist * 100:.3f}% | RSI={rsi_val:.1f} | ATR%={atr_pct*100:.3f}% | Vol>1.5x")
        return None

class StochCrossStrategy(BaseStrategy):
    id = "stoch_cross"; name = "Stochastic Zone Crossover"; timeframe = "1m"; leverage = 10; avg_signals_per_hour = 0.8
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 20: return None
        closes = [c.close for c in candles]; stoch_now = stochastic(candles, 14); stoch_prev = stochastic(candles[:-1], 14); rsi_val = rsi(closes, 14)
        if stoch_now["k"] < 25 and stoch_prev["k"] < stoch_prev["d"] and stoch_now["k"] > stoch_now["d"] and rsi_val < 40:
            conf = 0.59 + min((25 - stoch_now["k"]) / 100, 0.1)
            return Signal(side="LONG", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=clamp_leverage(10, CONFIG), reason=f"Stoch bullish cross in OS | K={stoch_now['k']:.1f} D={stoch_now['d']:.1f} | RSI={rsi_val:.1f}")
        if stoch_now["k"] > 75 and stoch_prev["k"] > stoch_prev["d"] and stoch_now["k"] < stoch_now["d"] and rsi_val > 60:
            conf = 0.59 + min((stoch_now["k"] - 75) / 100, 0.1)
            return Signal(side="SHORT", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=clamp_leverage(10, CONFIG), reason=f"Stoch bearish cross in OB | K={stoch_now['k']:.1f} D={stoch_now['d']:.1f} | RSI={rsi_val:.1f}")
        return None

class ATRBreakoutStrategy(BaseStrategy):
    id = "atr_breakout"; name = "ATR Volatility Breakout"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.6
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 30: return None
        closes = [c.close for c in candles]; price = closes[-1]
        atr_val = atr(candles, 14)
        if atr_val == 0: return None
        prev = candles[-2]; ema20 = ema(closes, 20)
        if ema20 is None: return None
        vol_ratio = volume_spike(candles, 15)
        if price > prev.high + atr_val * 0.5 and price > ema20 and vol_ratio > 1.0:
            conf = 0.60 + (0.08 if vol_ratio > 1.5 else 0) + (0.04 if price > ema20 else 0)
            return Signal(side="LONG", confidence=conf, tp_percent=0.010, sl_percent=0.005, leverage=8, reason=f"ATR breakout UP | Break={((price - prev.high) / atr_val):.2f}x ATR | Vol={vol_ratio:.1f}x")
        if price < prev.low - atr_val * 0.5 and price < ema20 and vol_ratio > 1.0:
            conf = 0.60 + (0.08 if vol_ratio > 1.5 else 0) + (0.04 if price < ema20 else 0)
            return Signal(side="SHORT", confidence=conf, tp_percent=0.010, sl_percent=0.005, leverage=8, reason=f"ATR breakout DOWN | Break={((prev.low - price) / atr_val):.2f}x ATR | Vol={vol_ratio:.1f}x")
        return None

class TripleEMAStrategy(BaseStrategy):
    id = "triple_ema"; name = "Triple EMA Ribbon Entry"; timeframe = "1m"; leverage = 12; avg_signals_per_hour = 0.8
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 30: return None
        closes = [c.close for c in candles]; ema5 = ema(closes, 5); ema13 = ema(closes, 13); ema21 = ema(closes, 21)
        if any(v is None for v in [ema5, ema13, ema21]): return None
        p_ema5 = ema(closes[:-1], 5); p_ema13 = ema(closes[:-1], 13)
        if p_ema5 is None or p_ema13 is None: return None
        rsi_val = rsi(closes, 14)
        bullish_now = ema5 > ema13 and ema13 > ema21; wasnt_bullish = p_ema5 <= p_ema13
        if bullish_now and wasnt_bullish and 45 < rsi_val < 72:
            conf = 0.61 + min((72 - rsi_val) / 200, 0.08)
            return Signal(side="LONG", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=12, reason=f"Triple EMA aligned bullish | 5>{ema5:.2f} 13>{ema13:.2f} 21>{ema21:.2f}")
        bearish_now = ema5 < ema13 and ema13 < ema21; wasnt_bearish = p_ema5 >= p_ema13
        if bearish_now and wasnt_bearish and 28 < rsi_val < 55:
            conf = 0.61 + min((rsi_val - 28) / 200, 0.08)
            return Signal(side="SHORT", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=12, reason=f"Triple EMA aligned bearish | 5<{ema5:.2f} 13<{ema13:.2f} 21<{ema21:.2f}")
        return None

class EngulfingSRStrategy(BaseStrategy):
    id = "engulfing_sr"; name = "Engulfing at Support/Resistance"; timeframe = "1m"; leverage = 10; avg_signals_per_hour = 0.6
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 40: return None
        last = candles[-1]; prev = candles[-2]
        bullish = (prev.close < prev.open and last.close > last.open and last.open <= prev.close and last.close >= prev.open)
        bearish = (prev.close > prev.open and last.close < last.open and last.open >= prev.close and last.close <= prev.open)
        if not bullish and not bearish: return None
        recent = candles[-40:]
        sorted_lows = sorted(c.low for c in recent); sorted_highs = sorted((c.high for c in recent), reverse=True)
        support = sorted_lows[3]; resistance = sorted_highs[3]
        near_support = abs(last.low - support) / support < 0.004 if support else False
        near_resistance = abs(last.high - resistance) / resistance < 0.004 if resistance else False
        if bullish and near_support:
            return Signal(side="LONG", confidence=0.63, tp_percent=0.006, sl_percent=0.003, leverage=clamp_leverage(10, CONFIG), reason=f"Bullish engulfing at support {support:.2f}")
        if bearish and near_resistance:
            return Signal(side="SHORT", confidence=0.63, tp_percent=0.006, sl_percent=0.003, leverage=clamp_leverage(10, CONFIG), reason=f"Bearish engulfing at resistance {resistance:.2f}")
        return None

class OBVDivergenceStrategy(BaseStrategy):
    id = "obv_divergence"; name = "OBV Divergence Reversal"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.5
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 40: return None
        closes = [c.close for c in candles]
        recent_candles = candles[-10:]; prior_candles = candles[-20:-10]
        recent_high = max(c.high for c in recent_candles); prior_high = max(c.high for c in prior_candles)
        recent_low = min(c.low for c in recent_candles); prior_low = min(c.low for c in prior_candles)
        obv_recent = obv(recent_candles); obv_prior = obv(prior_candles); rsi_val = rsi(closes, 14)
        if recent_high > prior_high and obv_recent < obv_prior and rsi_val > 55:
            conf = 0.60 + min((rsi_val - 55) / 200, 0.08)
            return Signal(side="SHORT", confidence=conf, tp_percent=0.007, sl_percent=0.004, leverage=8, reason=f"OBV bearish divergence | Price HH but OBV LH | RSI={rsi_val:.1f}")
        if recent_low < prior_low and obv_recent > obv_prior and rsi_val < 45:
            conf = 0.60 + min((45 - rsi_val) / 200, 0.08)
            return Signal(side="LONG", confidence=conf, tp_percent=0.007, sl_percent=0.004, leverage=8, reason=f"OBV bullish divergence | Price LL but OBV HL | RSI={rsi_val:.1f}")
        return None

class GaussianChannelStrategy(BaseStrategy):
    id = "gaussian_channel"; name = "Gaussian Channel Reversion"; timeframe = "1m"; leverage = 9; avg_signals_per_hour = 0.6
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 60: return None
        closes = [c.close for c in candles]
        price = closes[-1]
        gf_now = gaussian_filter(closes, 24)
        gf_prev = gaussian_filter(closes[:-1], 24)
        if gf_now is None or gf_prev is None:
            return None
        atr_val = atr(candles, 14)
        if not atr_val:
            return None
        width = max(price * 0.0012, atr_val * 1.35)
        upper = gf_now + width
        lower = gf_now - width
        prev_price = closes[-2]
        rsi_val = rsi(closes, 14)
        vol_ratio = volume_spike(candles, 20)

        # Mean-reversion at channel extremes
        if prev_price <= lower and price > lower and rsi_val < 46 and vol_ratio >= 0.8:
            conf = 0.61 + min((46 - rsi_val) / 140, 0.12) + (0.03 if vol_ratio > 1.2 else 0)
            return Signal(side="LONG", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=9, reason=f"Gaussian lower-band reclaim | RSI={rsi_val:.1f} | Vol={vol_ratio:.1f}x")
        if prev_price >= upper and price < upper and rsi_val > 54 and vol_ratio >= 0.8:
            conf = 0.61 + min((rsi_val - 54) / 140, 0.12) + (0.03 if vol_ratio > 1.2 else 0)
            return Signal(side="SHORT", confidence=conf, tp_percent=0.006, sl_percent=0.004, leverage=9, reason=f"Gaussian upper-band reject | RSI={rsi_val:.1f} | Vol={vol_ratio:.1f}x")
        return None

class TopDownAOIShiftStrategy(BaseStrategy):
    id = "topdown_aoi_shift"; name = "TopDown AOI Shift"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.35
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 260:
            return None
        closes = [c.close for c in candles]
        price = closes[-1]

        # Top-down direction proxy (higher TF respect)
        e50 = ema(closes, 50)
        e200 = ema(closes, 200)
        if e50 is None or e200 is None:
            return None
        bias_long = price > e200 and e50 > e200
        bias_short = price < e200 and e50 < e200
        if not (bias_long or bias_short):
            return None

        # AOI from frequently-tested local levels (last ~1-3 days proxy on 1m sample)
        look = candles[-240:]
        highs = [c.high for c in look]
        lows = [c.low for c in look]
        swing_high = max(highs)
        swing_low = min(lows)
        rng = max(1e-9, swing_high - swing_low)
        eq = (swing_high + swing_low) / 2.0

        # touch counting around candidate AOI bands
        band = rng * 0.01
        def touches(level: float) -> int:
            return sum(1 for c in look if abs(c.close - level) <= band)

        aoi_buy = swing_low + rng * 0.2
        aoi_sell = swing_high - rng * 0.2
        buy_touches = touches(aoi_buy)
        sell_touches = touches(aoi_sell)
        near_buy_aoi = abs(price - aoi_buy) <= band * 1.5
        near_sell_aoi = abs(price - aoi_sell) <= band * 1.5

        # execution: lower-timeframe shift of structure + reversal pattern
        c0, c1, c2 = candles[-1], candles[-2], candles[-3]
        bull_engulf = (c1.close < c1.open and c0.close > c0.open and c0.open <= c1.close and c0.close >= c1.open)
        bear_engulf = (c1.close > c1.open and c0.close < c0.open and c0.open >= c1.close and c0.close <= c1.open)
        bull_shift = c0.close > max(c1.high, c2.high)
        bear_shift = c0.close < min(c1.low, c2.low)

        vol = volume_spike(candles, 20)
        if vol < 0.8:
            return None

        atr_pct = (atr(candles, 14) / price) if price else 0.0
        sl = max(0.0035, min(0.008, atr_pct))
        tp = sl * 3.0

        if bias_long and near_buy_aoi and buy_touches >= 3 and price < eq and bull_engulf and bull_shift:
            conf = 0.65 + (0.03 if vol > 1.2 else 0.0)
            return Signal(side="LONG", confidence=min(conf, 0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"TopDown AOI long | touches={buy_touches} | shift+engulf | Vol={vol:.1f}x")

        if bias_short and near_sell_aoi and sell_touches >= 3 and price > eq and bear_engulf and bear_shift:
            conf = 0.65 + (0.03 if vol > 1.2 else 0.0)
            return Signal(side="SHORT", confidence=min(conf, 0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"TopDown AOI short | touches={sell_touches} | shift+engulf | Vol={vol:.1f}x")

        return None

class ImpulseMACDRegimeBreakoutStrategy(BaseStrategy):
    id = "impulse_macd_regime_breakout"; name = "Impulse MACD Regime Breakout"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.45
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 140:
            return None
        closes = [c.close for c in candles]
        price = closes[-1]

        # Build impulse proxy from MACD histogram normalized by ATR
        m_now = macd(closes)
        m_prev = macd(closes[:-1])
        hist_now = m_now["histogram"]
        hist_prev = m_prev["histogram"]
        hist_slope = hist_now - hist_prev

        atr_val = atr(candles, 14)
        if not atr_val or price <= 0:
            return None
        norm = max(1e-9, atr_val / price)

        # flat/range detector on recent impulse values
        imps = []
        for i in range(35, 0, -1):
            m = macd(closes[:-i])
            imps.append(m["histogram"] / norm)
        flat_ratio = sum(1 for x in imps[-20:] if abs(x) < 0.35) / 20.0
        was_flat = flat_ratio >= 0.65

        # extension levels (overbought/oversold style in impulse space)
        impulse_now = hist_now / norm
        ext_up = 1.1
        ext_dn = -1.1

        vol = volume_spike(candles, 20)
        if vol < 0.8:
            return None

        # breakout mode: after flat regime, require momentum pickup + price pressure
        c0, c1 = candles[-1], candles[-2]
        bullish_break = was_flat and impulse_now > 0.45 and hist_slope > 0 and c0.close > c1.high
        bearish_break = was_flat and impulse_now < -0.45 and hist_slope < 0 and c0.close < c1.low

        # extension-cross mode: only when outside extreme bands
        long_ext = (hist_prev / norm) <= ext_up and impulse_now > ext_up and hist_slope > 0
        short_ext = (hist_prev / norm) >= ext_dn and impulse_now < ext_dn and hist_slope < 0

        sl = max(0.0038, min(0.0075, norm * 0.9))
        tp = sl * 2.1

        if bullish_break or long_ext:
            mode = "breakout" if bullish_break else "extension"
            conf = 0.63 + (0.03 if was_flat else 0.0) + min(max(impulse_now, 0) / 8, 0.08)
            return Signal(side="LONG", confidence=min(conf, 0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"Impulse MACD long {mode} | imp={impulse_now:.2f} | flat={flat_ratio:.2f} | Vol={vol:.1f}x")

        if bearish_break or short_ext:
            mode = "breakout" if bearish_break else "extension"
            conf = 0.63 + (0.03 if was_flat else 0.0) + min(max(-impulse_now, 0) / 8, 0.08)
            return Signal(side="SHORT", confidence=min(conf, 0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"Impulse MACD short {mode} | imp={impulse_now:.2f} | flat={flat_ratio:.2f} | Vol={vol:.1f}x")

        return None

class EMA50BreakPullbackContinuationStrategy(BaseStrategy):
    id = "ema50_break_pullback_continuation"; name = "EMA50 Break Pullback Continuation"; timeframe = "1m"; leverage = 9; avg_signals_per_hour = 0.5
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 140:
            return None
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        price = closes[-1]

        e50 = ema(closes, 50)
        if e50 is None:
            return None

        c0, c1, c2, c3 = candles[-1], candles[-2], candles[-3], candles[-4]
        avg_body = sum(abs(c.close - c.open) for c in candles[-25:]) / 25.0
        breakout_body = abs(c2.close - c2.open)
        oversized_breakout = breakout_body > avg_body * 2.4

        # breakout candle across EMA50
        bull_break = c2.open < e50 and c2.close > e50
        bear_break = c2.open > e50 and c2.close < e50

        # pullback = at least 2 opposite candles after breakout
        bull_pullback_ok = (c1.close < c1.open and c0.close < c0.open)
        bear_pullback_ok = (c1.close > c1.open and c0.close > c0.open)

        # invalidation during pullback
        bull_invalid = min(c1.low, c0.low) < e50
        bear_invalid = max(c1.high, c0.high) > e50

        # pre-pullback swing levels (before c1,c0)
        swing_high = max(highs[-12:-2])
        swing_low = min(lows[-12:-2])

        vol = volume_spike(candles, 20)
        if vol < 0.8 or oversized_breakout:
            return None

        atr_val = atr(candles, 14)
        if not atr_val or price <= 0:
            return None
        atr_pct = atr_val / price
        sl = max(0.0038, min(0.0075, atr_pct * 1.1))
        tp = sl * 2.0

        # Need body close breakout of structure line on trigger candle
        long_trigger = c0.close > swing_high and c0.close > c0.open
        short_trigger = c0.close < swing_low and c0.close < c0.open

        if bull_break and bull_pullback_ok and (not bull_invalid) and long_trigger:
            conf = 0.64 + min((vol - 0.8) / 4, 0.08)
            return Signal(side="LONG", confidence=min(conf,0.90), tp_percent=tp, sl_percent=sl, leverage=9, reason=f"EMA50 break+pullback long | swing break | Vol={vol:.1f}x")

        if bear_break and bear_pullback_ok and (not bear_invalid) and short_trigger:
            conf = 0.64 + min((vol - 0.8) / 4, 0.08)
            return Signal(side="SHORT", confidence=min(conf,0.90), tp_percent=tp, sl_percent=sl, leverage=9, reason=f"EMA50 break+pullback short | swing break | Vol={vol:.1f}x")

        return None

class OrderBlockMTFInducementBreakerStrategy(BaseStrategy):
    id = "orderblock_mtf_inducement_breaker"; name = "OrderBlock MTF Inducement Breaker"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.4
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 260:
            return None
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        price = closes[-1]

        e50 = ema(closes, 50)
        e200 = ema(closes, 200)
        if e50 is None or e200 is None:
            return None
        bull_bias = price > e200 and e50 >= e200
        bear_bias = price < e200 and e50 <= e200
        if not (bull_bias or bear_bias):
            return None

        # HTF-like order block proxies from impulse bases in recent history
        look = candles[-220:]
        bull_obs = []
        bear_obs = []
        for i in range(8, len(look)-3):
            c = look[i]
            nxt = look[i+1]
            move = abs((nxt.close - c.close) / max(c.close, 1e-9))
            if c.close < c.open and nxt.close > c.high and move > 0.0012:
                bull_obs.append((c.low, c.high))
            if c.close > c.open and nxt.close < c.low and move > 0.0012:
                bear_obs.append((c.low, c.high))
        if not bull_obs and not bear_obs:
            return None

        # nearest active zones
        near_bull = min(bull_obs, key=lambda z: abs(price - ((z[0]+z[1])/2))) if bull_obs else None
        near_bear = min(bear_obs, key=lambda z: abs(price - ((z[0]+z[1])/2))) if bear_obs else None

        vol = volume_spike(candles, 20)
        if vol < 0.8:
            return None

        c0, c1, c2 = candles[-1], candles[-2], candles[-3]
        atr_val = atr(candles, 14)
        if not atr_val or price <= 0:
            return None
        sl = max(0.0036, min(0.008, (atr_val / price) * 1.0))

        # mode 1: MTF confirmation at OB
        if bull_bias and near_bull:
            zl, zh = near_bull
            in_zone = zl <= price <= zh * 1.002
            bull_confirm = (c1.close < c1.open and c0.close > c0.open and c0.close > c1.high)
            if in_zone and bull_confirm:
                conf = 0.64 + min((vol - 0.8) / 4, 0.06)
                return Signal(side="LONG", confidence=min(conf,0.90), tp_percent=sl*2.4, sl_percent=sl, leverage=8, reason=f"OB MTF long confirm | zone=[{zl:.4f},{zh:.4f}] | Vol={vol:.1f}x")

        if bear_bias and near_bear:
            zl, zh = near_bear
            in_zone = zl * 0.998 <= price <= zh
            bear_confirm = (c1.close > c1.open and c0.close < c0.open and c0.close < c1.low)
            if in_zone and bear_confirm:
                conf = 0.64 + min((vol - 0.8) / 4, 0.06)
                return Signal(side="SHORT", confidence=min(conf,0.90), tp_percent=sl*2.4, sl_percent=sl, leverage=8, reason=f"OB MTF short confirm | zone=[{zl:.4f},{zh:.4f}] | Vol={vol:.1f}x")

        # mode 2: inducement sweep trap near OB
        if bull_bias and near_bull:
            zl, zh = near_bull
            swept = c1.low < zl and c1.close > zl
            reclaim = c0.close > zh and c0.close > c0.open
            if swept and reclaim:
                conf = 0.66
                return Signal(side="LONG", confidence=conf, tp_percent=sl*2.8, sl_percent=sl, leverage=8, reason=f"OB inducement long sweep+reclaim | zone=[{zl:.4f},{zh:.4f}]")

        if bear_bias and near_bear:
            zl, zh = near_bear
            swept = c1.high > zh and c1.close < zh
            reject = c0.close < zl and c0.close < c0.open
            if swept and reject:
                conf = 0.66
                return Signal(side="SHORT", confidence=conf, tp_percent=sl*2.8, sl_percent=sl, leverage=8, reason=f"OB inducement short sweep+reject | zone=[{zl:.4f},{zh:.4f}]")

        # mode 3: breaker block retest (broken OB becomes opposite S/R)
        if near_bear and bull_bias:
            zl, zh = near_bear
            broken_up = closes[-6] < zl and closes[-3] > zh
            retest_hold = c1.low <= zh and c0.close > zh
            if broken_up and retest_hold:
                conf = 0.63
                return Signal(side="LONG", confidence=conf, tp_percent=sl*2.2, sl_percent=sl, leverage=8, reason=f"OB breaker long retest | old bear zone=[{zl:.4f},{zh:.4f}]")

        if near_bull and bear_bias:
            zl, zh = near_bull
            broken_dn = closes[-6] > zh and closes[-3] < zl
            retest_fail = c1.high >= zl and c0.close < zl
            if broken_dn and retest_fail:
                conf = 0.63
                return Signal(side="SHORT", confidence=conf, tp_percent=sl*2.2, sl_percent=sl, leverage=8, reason=f"OB breaker short retest | old bull zone=[{zl:.4f},{zh:.4f}]")

        return None

class FVGBOSInversionMTFStrategy(BaseStrategy):
    id = "fvg_bos_inversion_mtf"; name = "FVG BOS Inversion MTF"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.45
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 240:
            return None
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        price = closes[-1]

        # MTF trend proxy
        e200 = ema(closes, 200)
        if e200 is None:
            return None
        bull_bias = price > e200
        bear_bias = price < e200
        if not (bull_bias or bear_bias):
            return None

        vol = volume_spike(candles, 20)
        if vol < 0.8:
            return None

        # Find recent FVGs (3-candle imbalance)
        bull_fvgs = []  # (low_bound, high_bound, idx)
        bear_fvgs = []
        start = max(3, len(candles) - 120)
        for i in range(start, len(candles)-2):
            c1, c2, c3 = candles[i], candles[i+1], candles[i+2]
            if c1.high < c3.low:  # bullish gap
                bull_fvgs.append((c1.high, c3.low, i+2))
            if c1.low > c3.high:  # bearish gap
                bear_fvgs.append((c3.high, c1.low, i+2))

        if not bull_fvgs and not bear_fvgs:
            return None

        def unmitigated(fvg):
            lo, hi, idx = fvg
            for c in candles[idx+1:-1]:
                if c.low <= hi and c.high >= lo:
                    return False
            return True

        # nearest unmitigated FVG in-direction
        bull_candidates = [f for f in bull_fvgs if unmitigated(f)] if bull_bias else []
        bear_candidates = [f for f in bear_fvgs if unmitigated(f)] if bear_bias else []
        near_bull = min(bull_candidates, key=lambda f: abs(price - ((f[0]+f[1])/2))) if bull_candidates else None
        near_bear = min(bear_candidates, key=lambda f: abs(price - ((f[0]+f[1])/2))) if bear_candidates else None

        # BOS check (body close structure break)
        bos_up = closes[-1] > max(highs[-25:-1])
        bos_dn = closes[-1] < min(lows[-25:-1])

        atr_val = atr(candles, 14)
        if not atr_val or price <= 0:
            return None
        sl = max(0.0038, min(0.008, (atr_val / price) * 1.0))

        c0, c1 = candles[-1], candles[-2]

        # Mode 1: primary BOS + unmitigated FVG midpoint retest
        if near_bull and bull_bias:
            lo, hi, _ = near_bull
            mid = (lo + hi) / 2
            in_gap = lo <= price <= hi
            confirm = c0.close > c0.open and c0.close > c1.high
            if bos_up and in_gap and price >= mid and confirm:
                conf = 0.65 + min((vol - 0.8) / 4, 0.06)
                return Signal(side="LONG", confidence=min(conf, 0.90), tp_percent=sl*2.5, sl_percent=sl, leverage=8, reason=f"FVG BOS long | unmitigated mid-retest | gap=[{lo:.4f},{hi:.4f}]")

        if near_bear and bear_bias:
            lo, hi, _ = near_bear
            mid = (lo + hi) / 2
            in_gap = lo <= price <= hi
            confirm = c0.close < c0.open and c0.close < c1.low
            if bos_dn and in_gap and price <= mid and confirm:
                conf = 0.65 + min((vol - 0.8) / 4, 0.06)
                return Signal(side="SHORT", confidence=min(conf, 0.90), tp_percent=sl*2.5, sl_percent=sl, leverage=8, reason=f"FVG BOS short | unmitigated mid-retest | gap=[{lo:.4f},{hi:.4f}]")

        # Mode 2: inversion FVG (broken once, opposite retest)
        if near_bear and bull_bias:
            lo, hi, _ = near_bear
            broken_up = closes[-8] < lo and closes[-3] > hi
            retest_hold = c1.low <= hi and c0.close > hi
            if broken_up and retest_hold:
                return Signal(side="LONG", confidence=0.64, tp_percent=sl*2.2, sl_percent=sl, leverage=8, reason=f"FVG inversion long | broken bear gap retest [{lo:.4f},{hi:.4f}]")

        if near_bull and bear_bias:
            lo, hi, _ = near_bull
            broken_dn = closes[-8] > hi and closes[-3] < lo
            retest_fail = c1.high >= lo and c0.close < lo
            if broken_dn and retest_fail:
                return Signal(side="SHORT", confidence=0.64, tp_percent=sl*2.2, sl_percent=sl, leverage=8, reason=f"FVG inversion short | broken bull gap retest [{lo:.4f},{hi:.4f}]")

        return None

class VSAVolumeTruthStrategy(BaseStrategy):
    id = "vsa_volume_truth"
    name = "VSA Volume Truth (Effort vs Result)"
    timeframe = "1m"
    leverage = 8
    avg_signals_per_hour = 0.45

    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 80:
            return None

        closes = [c.close for c in candles]
        ema20 = ema(closes, 20)
        ema50 = ema(closes, 50)
        if ema20 is None or ema50 is None:
            return None

        # 1m execution + light HTF context proxy from recent 5-candle groups.
        c0, c1, c2 = candles[-1], candles[-2], candles[-3]
        vol_avg20 = sum(c.volume for c in candles[-21:-1]) / 20 if len(candles) >= 21 else 0
        if vol_avg20 <= 0:
            return None

        vol0 = c0.volume / vol_avg20
        vol1 = c1.volume / vol_avg20
        spread0 = max(c0.high - c0.low, c0.close * 0.0003)
        body0 = abs(c0.close - c0.open)
        body_ratio0 = body0 / spread0 if spread0 else 0

        up_trend = (ema20 > ema50 and c0.close > ema20)
        dn_trend = (ema20 < ema50 and c0.close < ema20)

        # --- VSA Pattern 1: Stopping Volume (reversal long) ---
        # Big effort (volume) + small result (narrow body) after decline.
        decline = c2.close > c1.close and c1.close >= c0.close
        stopping_vol = decline and vol0 >= 1.8 and body_ratio0 <= 0.35 and c0.close > c0.low + spread0 * 0.45
        if stopping_vol and not dn_trend:
            conf = 0.63 + min((vol0 - 1.8) / 4, 0.12)
            return Signal(
                side="LONG",
                confidence=min(conf, 0.86),
                tp_percent=0.0075,
                sl_percent=0.0038,
                leverage=8,
                reason=f"VSA stopping volume long | Vol={vol0:.1f}x | Body/Spread={body_ratio0:.2f}",
            )

        # --- VSA Pattern 2: Buying Climax (reversal short) ---
        rally = c2.close < c1.close and c1.close <= c0.close
        buying_climax = rally and vol0 >= 1.8 and body_ratio0 <= 0.35 and c0.close < c0.high - spread0 * 0.45
        if buying_climax and not up_trend:
            conf = 0.63 + min((vol0 - 1.8) / 4, 0.12)
            return Signal(
                side="SHORT",
                confidence=min(conf, 0.86),
                tp_percent=0.0075,
                sl_percent=0.0038,
                leverage=8,
                reason=f"VSA buying climax short | Vol={vol0:.1f}x | Body/Spread={body_ratio0:.2f}",
            )

        # --- VSA Pattern 3: No Supply / No Demand continuation ---
        # low-volume pullback candle followed by directional confirmation candle.
        low_vol_pullback = (vol1 <= 0.75)
        long_confirm = up_trend and low_vol_pullback and c1.close < c1.open and c0.close > c0.high - spread0 * 0.20
        short_confirm = dn_trend and low_vol_pullback and c1.close > c1.open and c0.close < c0.low + spread0 * 0.20

        if long_confirm:
            conf = 0.62 + min((0.9 - vol1), 0.08) + (0.04 if vol0 >= 1.0 else 0)
            return Signal(
                side="LONG",
                confidence=min(conf, 0.84),
                tp_percent=0.0065,
                sl_percent=0.0035,
                leverage=8,
                reason=f"VSA no-supply continuation long | PullbackVol={vol1:.2f}x | ConfirmVol={vol0:.2f}x",
            )

        if short_confirm:
            conf = 0.62 + min((0.9 - vol1), 0.08) + (0.04 if vol0 >= 1.0 else 0)
            return Signal(
                side="SHORT",
                confidence=min(conf, 0.84),
                tp_percent=0.0065,
                sl_percent=0.0035,
                leverage=8,
                reason=f"VSA no-demand continuation short | PullbackVol={vol1:.2f}x | ConfirmVol={vol0:.2f}x",
            )

        return None


class EMA20ZoneHeikinRSIStrategy(BaseStrategy):
    id = "ema20_zone_heikin_rsi"; name = "EMA20 Zone + Heikin RSI"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.55
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 90:
            return None
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        price = closes[-1]

        ehi = ema(highs, 20)
        elo = ema(lows, 20)
        ehi_p = ema(highs[:-1], 20)
        elo_p = ema(lows[:-1], 20)
        if any(v is None for v in [ehi, elo, ehi_p, elo_p]):
            return None

        # zone slope / flat filter
        slope_mid = ((ehi + elo) / 2) - ((ehi_p + elo_p) / 2)
        slope_pct = slope_mid / max(price, 1e-9)
        if abs(slope_pct) < 0.00025:
            return None
        up_zone = slope_pct > 0 and price > elo
        dn_zone = slope_pct < 0 and price < ehi

        # heikin-ashi candles (recent)
        ha = heikin_ashi(candles[-6:])
        if len(ha) < 3:
            return None
        h0, h1 = ha[-1], ha[-2]
        ha_green = h0['close'] > h0['open']
        ha_red = h0['close'] < h0['open']

        r = rsi(closes, 14)
        vol = volume_spike(candles, 20)
        if vol < 0.8:
            return None

        # trigger on HA color continuation/flip with RSI 50 confirmation
        long_ok = up_zone and ha_green and r > 50 and (h1['close'] <= h1['open'] or h0['close'] > h1['high'])
        short_ok = dn_zone and ha_red and r < 50 and (h1['close'] >= h1['open'] or h0['close'] < h1['low'])

        atr_val = atr(candles, 14)
        if not atr_val or price <= 0:
            return None
        sl = max(0.0035, min(0.0075, (atr_val / price) * 1.0))
        tp = sl * 3.0

        if long_ok:
            conf = 0.63 + min((r - 50) / 150, 0.08)
            return Signal(side="LONG", confidence=min(conf, 0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"EMA20 zone up + HA green + RSI>50 | RSI={r:.1f} | Vol={vol:.1f}x")
        if short_ok:
            conf = 0.63 + min((50 - r) / 150, 0.08)
            return Signal(side="SHORT", confidence=min(conf, 0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"EMA20 zone down + HA red + RSI<50 | RSI={r:.1f} | Vol={vol:.1f}x")
        return None

class FibPrecisionRespectMETStrategy(BaseStrategy):
    id = "fib_precision_respect_met"; name = "Fib Precision Respect MET"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.4
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 220:
            return None
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        price = closes[-1]

        # top-down directional proxy
        e200 = ema(closes, 200)
        if e200 is None:
            return None
        bull_bias = price > e200
        bear_bias = price < e200
        if not (bull_bias or bear_bias):
            return None

        # precision anchor: wick-to-wick swing range
        swing_high = max(highs[-120:])
        swing_low = min(lows[-120:])
        rng = max(1e-9, swing_high - swing_low)

        # fib levels from low->high (bull), high->low (bear)
        fib50 = swing_low + 0.5 * rng
        fib618 = swing_low + 0.618 * rng
        fib382 = swing_low + 0.382 * rng

        # golden respect rule: price hesitates/rejects at fib zone
        c0, c1, c2 = candles[-1], candles[-2], candles[-3]
        bull_reject = (c0.low <= fib618 <= c0.high or c0.low <= fib50 <= c0.high) and c0.close > c0.open and (c0.close > c1.high or (c0.high - c0.close) < (c0.close - c0.open))
        bear_reject = (c0.low <= fib618 <= c0.high or c0.low <= fib50 <= c0.high) and c0.close < c0.open and (c0.close < c1.low or (c0.close - c0.low) < (c0.open - c0.close))

        # avoid if near very recent extreme key-level collision (slice-through behavior)
        key_collision = abs(price - swing_high) / max(price,1e-9) < 0.0012 or abs(price - swing_low) / max(price,1e-9) < 0.0012
        if key_collision:
            return None

        vol = volume_spike(candles, 20)
        if vol < 0.8:
            return None

        atr_val = atr(candles, 14)
        if not atr_val or price <= 0:
            return None
        sl = max(0.0038, min(0.008, (atr_val / price) * 1.0))
        tp = sl * 2.8

        # time-zone style confluence proxy: recurring turn cadence from recent pivots
        pivot_turn = (c2.close < c1.close > c0.close) or (c2.close > c1.close < c0.close)
        tz_bonus = 0.02 if pivot_turn else 0.0

        if bull_bias and price >= fib382 and price <= fib618 and bull_reject:
            conf = 0.64 + min((vol - 0.8)/4, 0.06) + tz_bonus
            return Signal(side="LONG", confidence=min(conf,0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"Fib respect long | wick-anchor 50/61.8 reject | Vol={vol:.1f}x")

        if bear_bias and price <= (swing_high - 0.382 * rng) and price >= (swing_high - 0.618 * rng) and bear_reject:
            conf = 0.64 + min((vol - 0.8)/4, 0.06) + tz_bonus
            return Signal(side="SHORT", confidence=min(conf,0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"Fib respect short | wick-anchor 50/61.8 reject | Vol={vol:.1f}x")

        return None

class EMARSIStochTripwireStrategy(BaseStrategy):
    id = "ema_rsi_stoch_tripwire"; name = "EMA RSI Stoch Tripwire"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.5
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 170:
            return None
        closes = [c.close for c in candles]
        price = closes[-1]

        e25 = ema(closes, 25)
        e75 = ema(closes, 75)
        e140 = ema(closes, 140)
        e25p = ema(closes[:-1], 25)
        e75p = ema(closes[:-1], 75)
        e140p = ema(closes[:-1], 140)
        if any(v is None for v in [e25, e75, e140, e25p, e75p, e140p]):
            return None

        up_trend = price > e25 > e75 > e140 and e25 > e25p and e75 > e75p and e140 > e140p
        dn_trend = price < e25 < e75 < e140 and e25 < e25p and e75 < e75p and e140 < e140p
        if not (up_trend or dn_trend):
            return None

        r = rsi(closes, 75)
        if (up_trend and r <= 50) or (dn_trend and r >= 50):
            return None

        st = stochastic(candles, 14)
        st_p = stochastic(candles[:-1], 14)
        if st is None or st_p is None:
            return None
        k, d = st["k"], st["d"]
        kp, dp = st_p["k"], st_p["d"]

        # bounce zone around EMA tripwires
        near25 = abs(price - e25) / max(price, 1e-9) < 0.0025
        near75 = abs(price - e75) / max(price, 1e-9) < 0.0035
        near140 = abs(price - e140) / max(price, 1e-9) < 0.0045
        in_bounce_zone = near25 or near75 or near140

        vol = volume_spike(candles, 20)
        if vol < 0.8 or not in_bounce_zone:
            return None

        long_trigger = up_trend and kp <= dp and k > d and max(k, d) < 25
        short_trigger = dn_trend and kp >= dp and k < d and min(k, d) > 75

        atr_val = atr(candles, 14)
        if not atr_val or price <= 0:
            return None
        sl = max(0.0038, min(0.008, (atr_val / price) * 1.0))
        tp = sl * 3.0

        if long_trigger:
            conf = 0.64 + min((r - 50) / 180, 0.08)
            return Signal(side="LONG", confidence=min(conf, 0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"EMA tripwire long | RSI75={r:.1f} | Stoch up-cross | Vol={vol:.1f}x")
        if short_trigger:
            conf = 0.64 + min((50 - r) / 180, 0.08)
            return Signal(side="SHORT", confidence=min(conf, 0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"EMA tripwire short | RSI75={r:.1f} | Stoch down-cross | Vol={vol:.1f}x")
        return None

class UTBotATRAdaptiveTrendStrategy(BaseStrategy):
    id = "utbot_atr_adaptive_trend"; name = "UT Bot ATR Adaptive Trend"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.55
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 120:
            return None
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        price = closes[-1]

        atr_val = atr(candles, 14)
        if not atr_val or price <= 0:
            return None

        # UT-style trailing stop approximation
        sens = 1.8
        nloss = atr_val * sens
        trail = closes[-2] - nloss
        for c in closes[-20:-1]:
            trail = max(trail, c - nloss)
        trail_short = closes[-2] + nloss
        for c in closes[-20:-1]:
            trail_short = min(trail_short, c + nloss)

        prev_close = closes[-2]
        long_cross = prev_close <= trail and price > trail
        short_cross = prev_close >= trail_short and price < trail_short

        # trend filter to reduce chop
        e50 = ema(closes, 50)
        e50p = ema(closes[:-1], 50)
        if e50 is None or e50p is None:
            return None
        up_trend = price > e50 and e50 > e50p
        dn_trend = price < e50 and e50 < e50p

        # volatility-adaptive body strength filter
        body = abs(candles[-1].close - candles[-1].open)
        avg_body = sum(abs(c.close - c.open) for c in candles[-20:]) / 20.0
        if body < avg_body * 0.7:
            return None

        vol = volume_spike(candles, 20)
        if vol < 0.8:
            return None

        sl = max(0.0038, min(0.008, (atr_val / price) * 1.1))
        tp = sl * 2.4

        if long_cross and up_trend:
            conf = 0.64 + min((vol - 0.8)/4, 0.07)
            return Signal(side="LONG", confidence=min(conf,0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"UT ATR long cross | trail reclaim | Vol={vol:.1f}x")
        if short_cross and dn_trend:
            conf = 0.64 + min((vol - 0.8)/4, 0.07)
            return Signal(side="SHORT", confidence=min(conf,0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"UT ATR short cross | trail reject | Vol={vol:.1f}x")
        return None

class MASlopeCrossoverSRStrategy(BaseStrategy):
    id = "ma_slope_crossover_sr"; name = "MA Slope Crossover SR"; timeframe = "1m"; leverage = 8; avg_signals_per_hour = 0.55
    def evaluate(self, candles: list[Candle]) -> Optional[Signal]:
        if len(candles) < 230:
            return None
        closes = [c.close for c in candles]
        price = closes[-1]

        ma_fast = ema(closes, 50)
        ma_slow = ema(closes, 200)
        ma_fast_p = ema(closes[:-1], 50)
        ma_slow_p = ema(closes[:-1], 200)
        if any(v is None for v in [ma_fast, ma_slow, ma_fast_p, ma_slow_p]):
            return None

        slope_fast = (ma_fast - ma_fast_p) / max(price, 1e-9)
        slope_slow = (ma_slow - ma_slow_p) / max(price, 1e-9)
        if abs(slope_fast) < 0.00015 and abs(slope_slow) < 0.00008:
            return None  # sideways filter

        up_trend = price > ma_fast > ma_slow and slope_fast > 0 and slope_slow >= 0
        dn_trend = price < ma_fast < ma_slow and slope_fast < 0 and slope_slow <= 0

        # dynamic SR pullback near fast MA
        near_fast = abs(price - ma_fast) / max(price, 1e-9) < 0.0025
        c0, c1 = candles[-1], candles[-2]
        pullback_long_confirm = near_fast and c1.close < c1.open and c0.close > c0.open and c0.close > c1.high
        pullback_short_confirm = near_fast and c1.close > c1.open and c0.close < c0.open and c0.close < c1.low

        # crossover reversal mode
        bull_cross = ma_fast_p <= ma_slow_p and ma_fast > ma_slow
        bear_cross = ma_fast_p >= ma_slow_p and ma_fast < ma_slow

        vol = volume_spike(candles, 20)
        if vol < 0.8:
            return None

        atr_val = atr(candles, 14)
        if not atr_val or price <= 0:
            return None
        sl = max(0.0038, min(0.008, (atr_val / price) * 1.0))
        tp = sl * 2.4

        if (up_trend and pullback_long_confirm) or (bull_cross and slope_fast > 0):
            mode = "pullback" if (up_trend and pullback_long_confirm) else "crossover"
            conf = 0.63 + min((vol - 0.8)/4, 0.07)
            return Signal(side="LONG", confidence=min(conf, 0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"MA long {mode} | fast/slow aligned | Vol={vol:.1f}x")

        if (dn_trend and pullback_short_confirm) or (bear_cross and slope_fast < 0):
            mode = "pullback" if (dn_trend and pullback_short_confirm) else "crossover"
            conf = 0.63 + min((vol - 0.8)/4, 0.07)
            return Signal(side="SHORT", confidence=min(conf, 0.90), tp_percent=tp, sl_percent=sl, leverage=8, reason=f"MA short {mode} | fast/slow aligned | Vol={vol:.1f}x")

        return None

# ── BUG-006 FIX: FundingFadeStrategy() removed from ALL_STRATEGIES ──
ALL_STRATEGIES = [
    EMAScalpStrategy(), RSISnapStrategy(), MACDFlipStrategy(), MACDMoneyMapTrendStrategy(), MACDMoneyMapReversalStrategy(), EMACCIMACDComboStrategy(), EMARibbon33889PullbackStrategy(), EMA1020CCIMomentumStrategy(), PriceActionDLEStrategy(), TopDownAOIShiftStrategy(), ImpulseMACDRegimeBreakoutStrategy(), EMA50BreakPullbackContinuationStrategy(), OrderBlockMTFInducementBreakerStrategy(), FVGBOSInversionMTFStrategy(), VSAVolumeTruthStrategy(), FibPrecisionRespectMETStrategy(), EMARSIStochTripwireStrategy(), UTBotATRAdaptiveTrendStrategy(), MASlopeCrossoverSRStrategy(), VWAPBounceStrategy(), StochCrossStrategy(), ATRBreakoutStrategy(), TripleEMAStrategy(), EngulfingSRStrategy(), OBVDivergenceStrategy(), GaussianChannelStrategy()
]



def load_2h_strategies():
    try:
        from new_strategies_2h import NEW_2H_STRATEGIES, NEW_CATEGORIES
    except Exception as e:
        logger.warning(f"Failed to load 2h strategies: {e}")
        return
    existing = {s.id for s in ALL_STRATEGIES}
    for s in NEW_2H_STRATEGIES:
        if s.id not in existing:
            ALL_STRATEGIES.append(s)
    CONFIG["strategy_categories"].update(NEW_CATEGORIES)

class CorrelationFilter:
    def __init__(self):
        self.pair_cooldowns: dict[str, float] = {}
        self.peak_balance: float = 0.0
        self.recent_signal_log: list[dict] = []
    def reset(self):
        self.pair_cooldowns.clear(); self.peak_balance = 0.0; self.recent_signal_log.clear()
    @staticmethod
    def get_strategy_category(strategy_id: str) -> str:
        for category, ids in CONFIG["strategy_categories"].items():
            if strategy_id in ids: return category
        return "unknown"
    def get_state(self) -> dict:
        now = time.time(); cooldown_sec = CONFIG["correlation"]["cooldown_sec"]; window_sec = CONFIG["correlation"]["same_side_window_sec"]
        return {
            "active_cooldowns": [{"pair": pair, "expires_in": cooldown_sec - (now - ts)} for pair, ts in self.pair_cooldowns.items() if now - ts < cooldown_sec],
            "recent_signal_count": sum(1 for s in self.recent_signal_log if now - s["timestamp"] < window_sec),
            "peak_balance": self.peak_balance,
        }
    @staticmethod
    def detect_btc_macro_move(market_data: dict[str, list[Candle]]) -> BTCMacroStatus:
        btc_candles = market_data.get("BTCUSDT"); lookback = CONFIG["correlation"]["btc_move_lookback"]
        if not btc_candles or len(btc_candles) < lookback + 1:
            return BTCMacroStatus(is_macro_move=False, direction=None, magnitude=0.0)
        recent = btc_candles[-lookback:]; price_now = recent[-1].close; price_then = recent[0].open
        move = (price_now - price_then) / price_then if price_then else 0
        return BTCMacroStatus(is_macro_move=abs(move) >= CONFIG["correlation"]["btc_move_threshold"], direction="LONG" if move > 0 else "SHORT", magnitude=abs(move), btc_price=price_now)
    def apply_same_side_flood_filter(self, signals: list[TradeSignal]) -> list[TradeSignal]:
        now = time.time(); window = CONFIG["correlation"]["same_side_window_sec"]; max_same = CONFIG["correlation"]["max_same_side_signals"]
        self.recent_signal_log = [s for s in self.recent_signal_log if now - s["timestamp"] < window]
        side_counts = {"LONG": 0, "SHORT": 0}
        side_blockers: dict[str, list[str]] = {"LONG": [], "SHORT": []}
        for s in self.recent_signal_log:
            sd = s.get("side")
            side_counts[sd] = side_counts.get(sd, 0) + 1
            sid = s.get("strategy_id") or ""
            if sid:
                side_blockers.setdefault(sd, []).append(sid)
        passed = []
        for sig in signals:
            if side_counts.get(sig.side, 0) >= max_same:
                blockers = [b for b in side_blockers.get(sig.side, []) if b]
                if blockers:
                    uniq = list(dict.fromkeys(blockers))
                    sig._blocked_by = ",".join(uniq[:5])
                    sig._filter_reason = f"Same-side flood: {side_counts[sig.side]} {sig.side}s already in window | blocked_by={sig._blocked_by}"
                else:
                    sig._filter_reason = f"Same-side flood: {side_counts[sig.side]} {sig.side}s already in window"
                sig._filtered = True
                continue
            side_counts[sig.side] = side_counts.get(sig.side, 0) + 1
            side_blockers.setdefault(sig.side, []).append(sig.strategy_id)
            passed.append(sig)
        return passed
    def apply_category_filter(self, signals: list[TradeSignal], active_trades: list[ActiveTrade]) -> list[TradeSignal]:
        max_per_cat = CONFIG["correlation"]["max_per_category"]
        active_cat_counts: dict[str, int] = {}
        active_blockers: dict[str, list[str]] = {}
        for trade in active_trades:
            cat = self.get_strategy_category(trade.strategy_id)
            active_cat_counts[cat] = active_cat_counts.get(cat, 0) + 1
            if getattr(trade, 'strategy_id', ''):
                active_blockers.setdefault(cat, []).append(trade.strategy_id)
        pending_cat_counts: dict[str, int] = {}
        pending_blockers: dict[str, list[str]] = {}
        passed = []
        for sig in signals:
            cat = sig.strategy_category
            total = active_cat_counts.get(cat, 0) + pending_cat_counts.get(cat, 0)
            if total >= max_per_cat:
                blockers = active_blockers.get(cat, []) + pending_blockers.get(cat, [])
                if blockers:
                    uniq = list(dict.fromkeys([b for b in blockers if b]))
                    sig._blocked_by = ",".join(uniq[:5])
                    sig._filter_reason = f"Category cap: {cat} has {total}/{max_per_cat} slots | blocked_by={sig._blocked_by}"
                else:
                    sig._filter_reason = f"Category cap: {cat} has {total}/{max_per_cat} slots"
                sig._filtered = True
                continue
            pending_cat_counts[cat] = pending_cat_counts.get(cat, 0) + 1
            pending_blockers.setdefault(cat, []).append(sig.strategy_id)
            passed.append(sig)
        return passed
    def apply_cooldown_filter(self, signals: list[TradeSignal]) -> list[TradeSignal]:
        now = time.time(); cooldown = CONFIG["correlation"]["cooldown_sec"]; passed = []
        for sig in signals:
            last = self.pair_cooldowns.get(sig.pair)
            if last and now - last < cooldown:
                sig._filtered = True; sig._filter_reason = f"Cooldown: {sig.pair} signaled {now - last:.0f}s ago"; continue
            passed.append(sig)
        return passed
    def check_drawdown_breaker(self, balance: float) -> DrawdownStatus:
        if balance > self.peak_balance: self.peak_balance = balance
        peak = self.peak_balance
        if peak == 0: return DrawdownStatus(paused=False, drawdown=0.0)
        drawdown = (peak - balance) / peak; threshold = CONFIG["correlation"]["max_drawdown_pause"]
        paused = drawdown >= threshold
        return DrawdownStatus(paused=paused, drawdown=drawdown, peak=peak, threshold=threshold, message=(f"CIRCUIT BREAKER: {drawdown * 100:.2f}% drawdown from peak ${peak:.2f}. New entries paused." if paused else None))
    def record_signals(self, signals: list[TradeSignal]):
        now = time.time()
        for sig in signals:
            self.pair_cooldowns[sig.pair] = now
            self.recent_signal_log.append({"side": sig.side, "strategy_id": sig.strategy_id, "timestamp": now})
    def set_pair_cooldown(self, pair: str, ts: float | None = None, strategy_id: str = ""):
        k = cooldown_key(pair, strategy_id) if strategy_id else pair
        self.pair_cooldowns[k] = ts or time.time()

correlation_filter = CorrelationFilter()

# Strategy-level cooldowns (across all pairs)
_strategy_cooldowns_until: dict[str, float] = {}
_STRATEGY_COOLDOWN_PATH = "/opt/multi-strat-engine/reports/strategy_cooldowns.json"


def _load_strategy_cooldowns():
    global _strategy_cooldowns_until
    try:
        import json
        from pathlib import Path
        p = Path(_STRATEGY_COOLDOWN_PATH)
        if p.exists():
            data = json.loads(p.read_text() or "{}")
            if isinstance(data, dict):
                _strategy_cooldowns_until = {str(k): float(v or 0) for k, v in data.items()}
    except Exception:
        pass


def _save_strategy_cooldowns():
    try:
        import json
        from pathlib import Path
        p = Path(_STRATEGY_COOLDOWN_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        # persist only active/future cooldowns
        now = time.time()
        active = {k: v for k, v in _strategy_cooldowns_until.items() if float(v) > now}
        p.write_text(json.dumps(active))
    except Exception:
        pass


def set_pair_cooldown(pair: str, ts: float | None = None, strategy_id: str = ""):
    correlation_filter.set_pair_cooldown(pair, ts, strategy_id=strategy_id)


def set_strategy_cooldown(strategy_id: str, seconds: float):
    if not strategy_id or not seconds:
        return
    _strategy_cooldowns_until[strategy_id] = time.time() + float(seconds)
    _save_strategy_cooldowns()


def get_strategy_cooldown_remaining(strategy_id: str) -> float:
    until = _strategy_cooldowns_until.get(strategy_id, 0.0)
    rem = max(0.0, float(until) - time.time())
    return rem

_load_strategy_cooldowns()

def run_signal_scan(market_data: dict[str, list[Candle]], active_trades: list[ActiveTrade], balance: float, funding: dict[str, float] | None = None, open_interest: dict[str, float] | None = None, spread_map: dict[str, tuple[float,float]] | None = None, strategies: list[BaseStrategy] | None = None, news_bias: dict[str, float] | None = None, market_data_5m: dict[str, list[Candle]] | None = None, market_data_15m: dict[str, list[Candle]] | None = None) -> ScanResult:
    if strategies is None: strategies = ALL_STRATEGIES
    result = ScanResult(); cfg = CONFIG
    if open_interest is None:
        open_interest = {}
    if spread_map is None:
        spread_map = {}
    if funding is None:
        funding = {}
    if news_bias is None:
        news_bias = {}
    if market_data_5m is None:
        market_data_5m = {}
    if market_data_15m is None:
        market_data_15m = {}

    drawdown_check = correlation_filter.check_drawdown_breaker(balance); result.drawdown = drawdown_check
    if drawdown_check.paused:
        result.diagnostics = ScanDiagnostics(raw_count=0, final=0, reason="CIRCUIT_BREAKER"); return result
    btc_macro = CorrelationFilter.detect_btc_macro_move(market_data); result.btc_macro = btc_macro
    max_per_pair = max(1, int(cfg.get("max_trades_per_pair", 1)))
    open_pair_counts: dict[str, int] = {}
    for t in active_trades:
        open_pair_counts[t.pair] = open_pair_counts.get(t.pair, 0) + 1
    slots_available = cfg["max_concurrent_trades"] - len(active_trades)
    max_1m = cfg.get("max_1m_trades", cfg["max_concurrent_trades"])
    max_2h = cfg.get("max_2h_trades", cfg["max_concurrent_trades"])
    active_1m = sum(1 for t in active_trades if not is_2h_strategy(t.strategy_id))
    active_2h = sum(1 for t in active_trades if is_2h_strategy(t.strategy_id))
    if slots_available <= 0:
        result.diagnostics = ScanDiagnostics(raw_count=0, final=0, reason="ALL_SLOTS_FULL"); return result
    raw_signals: list[TradeSignal] = []
    pair_regimes = {}
    base_conf_th = cfg.get("confidence_threshold", 0.55)
    base_confirm = cfg.get("confirm_signal", False)
    base_min_vol = cfg.get("min_volatility_pct", 0.0)
    ema_fast_n = cfg.get("trend_ema_fast", 0)
    ema_slow_n = cfg.get("trend_ema_slow", 0)
    regime_min_trend = cfg.get("regime_min_trend_pct", 0.0)
    atr_period = cfg.get("atr_period", 14)
    vol_target = cfg.get("vol_target_pct", 0.0)
    max_funding_long = cfg.get("max_funding_long", 0.0)
    max_funding_short = cfg.get("max_funding_short", 0.0)
    universe_1m = set(cfg.get("pairs", []))
    universe_2h = set(cfg.get("pairs_2h", cfg.get("pairs", [])))
    pair_universe = sorted(universe_1m | universe_2h)
    for pair in pair_universe:
        candles = market_data.get(pair)
        if not candles or len(candles) < 50: continue
        candles_5m = market_data_5m.get(pair)
        candles_15m = market_data_15m.get(pair)
        if detect_market_regime:
            try:
                pair_regimes[pair] = detect_market_regime(candles, candles_5m, candles_15m)
            except Exception:
                pair_regimes[pair] = None
        if open_pair_counts.get(pair, 0) >= max_per_pair: continue
        if btc_macro.is_macro_move and pair != "BTCUSDT": continue
        price = candles[-1].close
        # Trend filter (EMA)
        closes = [c.close for c in candles]
        ap = get_adaptive_params(pair, candles) if cfg.get("adaptive_params", {}).get("enabled", False) else None
        af = ap.get("ema_fast") if ap else ema_fast_n
        aslow = ap.get("ema_slow") if ap else ema_slow_n
        ema_fast = ema(closes, af) if af else None
        ema_slow = ema(closes, aslow) if aslow else None
        # Regime filter: if trend strength small, only allow reversion/structural
        trend_strength = 0.0
        if ema_fast is not None and ema_slow is not None and price:
            trend_strength = abs(ema_fast - ema_slow) / price
        for strategy in strategies:
            # strategy-level cooldown across all pairs
            if get_strategy_cooldown_remaining(strategy.id) > 0:
                continue
            is2h = is_2h_strategy(strategy.id)
            if is2h and pair not in universe_2h:
                continue
            if (not is2h) and pair not in universe_1m:
                continue
            # bb_squeeze uses 5m candles and 15m EMA50/100 trend filter
            use_candles = candles_5m if strategy.id == 'bb_squeeze' and candles_5m else candles
            if is2h and active_2h >= max_2h: continue
            if (not is2h) and active_1m >= max_1m: continue
            s_cfg = CONFIG_2H if is2h else {}
            conf_th = s_cfg.get("confidence_threshold", cfg.get("confidence_threshold_1m", base_conf_th)) if not is2h else s_cfg.get("confidence_threshold", base_conf_th)
            confirm = s_cfg.get("confirm_signal", base_confirm)
            min_vol = s_cfg.get("min_volatility_pct", base_min_vol)
            max_funding_long = s_cfg.get("max_funding_long", cfg.get("max_funding_long", 0.0))
            max_funding_short = s_cfg.get("max_funding_short", cfg.get("max_funding_short", 0.0))
            skip_trend = s_cfg.get("skip_trend_ema_filter", False)
            skip_regime = s_cfg.get("skip_regime_filter", False)
            price_used = use_candles[-1].close if use_candles else price
            # volatility filter (tier-specific)
            if min_vol and len(use_candles) >= 20:
                hi = max(c.high for c in use_candles[-20:])
                lo = min(c.low for c in use_candles[-20:])
                vol_pct = (hi - lo) / price_used if price_used else 0
                if vol_pct < min_vol:
                    continue
            if btc_macro.is_macro_move and pair == "BTCUSDT":
                cat = CorrelationFilter.get_strategy_category(strategy.id)
                if cat != "reversion": continue
            if hasattr(strategy, "needs_funding") and strategy.needs_funding:
                eval_result = strategy.evaluate(use_candles, funding_rate=funding.get(pair, 0.0))
            else:
                eval_result = strategy.evaluate(use_candles)
            if eval_result is None: continue
            # pair-specific strategy allowlist
            pf = cfg.get("pair_filters", {})
            if pair in pf and strategy.id not in pf[pair]:
                # keep selected global overlays available even on filtered pairs
                if strategy.id not in ("macd_money_map_trend", "macd_money_map_reversal", "ema_cci_macd_combo", "ema_ribbon_33889_pullback", "ema10_20_cci_momentum", "price_action_dle", "topdown_aoi_shift", "impulse_macd_regime_breakout", "ema50_break_pullback_continuation", "orderblock_mtf_inducement_breaker", "fvg_bos_inversion_mtf", "ema20_zone_heikin_rsi", "fib_precision_respect_met", "ema_rsi_stoch_tripwire", "utbot_atr_adaptive_trend", "ma_slope_crossover_sr", "vsa_volume_truth"):
                    continue
            bias = news_bias.get(pair, 0.0)
            adj_conf = max(0.0, min(1.0, eval_result.confidence + bias))
            if adj_conf < conf_th: continue
            if confirm and not is2h:
                confirm_lookback = int(cfg.get("confirm_lookback_1m", 2) or 2)
                if confirm_lookback > 1:
                    prev_eval = strategy.evaluate(use_candles[:-1]) if len(use_candles) > 51 else None
                    if not prev_eval or prev_eval.side != eval_result.side or prev_eval.confidence < conf_th:
                        continue
            # Regime filter
            if (not skip_regime) and regime_min_trend and trend_strength < regime_min_trend:
                if strategy.id in CONFIG["strategy_categories"].get("trend", []):
                    continue
            # Trend alignment (override for bb_squeeze using 15m EMA50/100)
            if strategy.id == 'bb_squeeze':
                if not candles_15m or len(candles_15m) < 120:
                    continue
                closes_15m = [c.close for c in candles_15m]
                ema50 = ema(closes_15m, 50)
                ema100 = ema(closes_15m, 100)
                if ema50 is None or ema100 is None:
                    continue
                price_15m = closes_15m[-1]
                if eval_result.side == "LONG" and not (price_15m > ema50 and ema50 > ema100):
                    continue
                if eval_result.side == "SHORT" and not (price_15m < ema50 and ema50 < ema100):
                    continue
            else:
                # Trend alignment (default)
                if (not skip_trend) and ema_fast is not None and ema_slow is not None:
                    if eval_result.side == "LONG" and not (price > ema_fast and ema_fast > ema_slow):
                        continue
                    if eval_result.side == "SHORT" and not (price < ema_fast and ema_fast < ema_slow):
                        continue
            # Funding filter
            if funding:
                fr = funding.get(pair, 0.0)
                if eval_result.side == "LONG" and max_funding_long and fr > max_funding_long:
                    continue
                if eval_result.side == "SHORT" and max_funding_short and fr < -max_funding_short:
                    continue
            # Volatility-targeted sizing (ATR)
            atr_val = atr(use_candles, atr_period) if atr_period else 0.0
            atr_pct = (atr_val / price_used) if price_used else 0.0
            min_size = float(cfg.get("min_trade_size_1m", cfg.get("min_trade_size", 10))) if not is2h else float(cfg.get("min_trade_size", 10))
            base_size = max(min_size, min(balance * cfg["risk_per_trade"], balance / cfg["max_concurrent_trades"]))
            if vol_target and atr_pct > 0:
                scale = vol_target / atr_pct
                trade_size = max(min_size, min(base_size * scale, base_size * 2))
            else:
                trade_size = base_size
            # Per-strategy / per-tier base USD cap (if configured)
            strat_cap = float((cfg.get("strategy_max_base_usd", {}) or {}).get(strategy.id, 0) or 0)
            if strat_cap > 0:
                trade_size = min(trade_size, strat_cap)
            if not is2h:
                cap_1m = float(cfg.get("max_base_margin_1m", 0) or 0)
                if cap_1m > 0:
                    trade_size = min(trade_size, cap_1m)
            if trade_size < min_size or balance < trade_size: continue
            tp_mult = s_cfg.get("tp_multiplier", cfg.get("tp_multiplier", 1.0))
            min_tp = s_cfg.get("min_tp_percent", cfg.get("min_tp_percent", 0.0))
            min_sl = s_cfg.get("min_sl_percent", cfg.get("min_sl_percent", 0.0))
            tp_pct = max(eval_result.tp_percent * tp_mult, min_tp)
            tp_price = price_used * (1 + tp_pct) if eval_result.side == "LONG" else price_used * (1 - tp_pct)
            sl_pct = max(eval_result.sl_percent, min_sl)
            sl_price = price_used * (1 - sl_pct) if eval_result.side == "LONG" else price_used * (1 + sl_pct)
            lev_cfg = CONFIG if is2h else {"max_leverage": cfg.get("max_leverage_1m", cfg.get("max_leverage", eval_result.leverage))}
            lev_used = clamp_leverage(eval_result.leverage, lev_cfg)
            economics = calculate_trade_economics(price_used, tp_price, sl_price, eval_result.side, trade_size, lev_used)
            if not economics.is_profitable: continue
            # min risk-reward guard
            min_rr = s_cfg.get("min_risk_reward", cfg.get("min_risk_reward", None))
            if min_rr is not None and economics.risk_reward < min_rr:
                continue
            reason = eval_result.reason
            if bias:
                reason = f"{reason} | news_bias={bias:+.3f}"
            raw_signals.append(TradeSignal(pair=pair, strategy_id=strategy.id, strategy_name=strategy.name, strategy_category=CorrelationFilter.get_strategy_category(strategy.id), side=eval_result.side, confidence=adj_conf, entry_price=price, tp_price=tp_price, sl_price=sl_price, leverage=lev_used, trade_size=trade_size, reason=reason, economics=economics))
    # apply market divergence boost (confidence only)
    div_cfg = cfg.get("divergence_boost", {})
    div_dir, div_count = (0,0)
    if div_cfg.get("enabled", False):
        div_dir, div_count = compute_market_divergence(market_data, div_cfg.get("lookback", 20))
    if div_dir != 0 and div_count >= div_cfg.get("pairs_min", 3):
        boost = div_cfg.get("boost", 0.04)
        for s in raw_signals:
            if (div_dir == 1 and s.side == "LONG") or (div_dir == -1 and s.side == "SHORT"):
                s.confidence += boost
    # apply OI context boost (confidence only)
    oi_cfg = cfg.get("oi_boost", {})
    if oi_cfg.get("enabled", False) and open_interest:
        for s in raw_signals:
            oi = open_interest.get(s.pair, 0.0)
            ctx = compute_oi_context(s.pair, s.entry_price, oi)
            if ctx in (1,2) and s.side == "LONG":
                s.confidence += oi_cfg.get("boost", 0.06)
    # apply agreement boost (same pair + same side)
    ab = cfg.get("agreement_boost", {})
    if ab.get("enabled", False):
        from collections import defaultdict
        counts = defaultdict(int)
        for s in raw_signals:
            counts[(s.pair, s.side)] += 1
        for s in raw_signals:
            if counts[(s.pair, s.side)] >= ab.get("min_strategies", 2):
                s.confidence += ab.get("boost", 0.05)

    # Market-regime activation/deactivation policy (moderate thresholds)
    if should_activate_strategy:
        gated = []
        for s in raw_signals:
            rg = pair_regimes.get(s.pair)
            if rg is None:
                gated.append(s)
                continue
            try:
                ok, why = should_activate_strategy(s.strategy_id, s.strategy_category, rg, pair=s.pair)
            except Exception:
                ok, why = True, "policy_error_fallback"
            if ok:
                gated.append(s)
            else:
                s._filtered = True
                s._filter_reason = f"activation_policy:{why}"
        raw_signals = gated

    raw_signals.sort(key=lambda s: s.confidence, reverse=True)



    pair_seen_counts: dict[str, int] = {}
    deduped: list[TradeSignal] = []
    for sig in raw_signals:
        allowed_for_pair = max_per_pair - open_pair_counts.get(sig.pair, 0)
        if allowed_for_pair <= 0:
            continue
        cur = pair_seen_counts.get(sig.pair, 0)
        if cur >= allowed_for_pair:
            continue
        pair_seen_counts[sig.pair] = cur + 1
        deduped.append(sig)
    diag = ScanDiagnostics(raw_count=len(deduped))
    after_cooldown = apply_cooldown_tiered(deduped); diag.after_cooldown = len(after_cooldown)
    after_flood = correlation_filter.apply_same_side_flood_filter(after_cooldown); diag.after_flood = len(after_flood)
    after_category = correlation_filter.apply_category_filter(after_flood, active_trades); diag.after_category = len(after_category)
    final_signals = after_category[:slots_available]; diag.final = len(final_signals)
    if slots_available < len(after_category):
        blockers = [s.strategy_id for s in final_signals][:5]
        for s in after_category[slots_available:]:
            s._filtered = True
            if blockers:
                s._blocked_by = ",".join(blockers)
                s._filter_reason = f"slot_limit | blocked_by={s._blocked_by}"
            else:
                s._filter_reason = "slot_limit"
    # NOTE: cooldown/same-side history is now set only on successful ENTRY in trade_loop.
    all_filtered = [s for s in deduped + after_cooldown + after_flood + after_category if s._filtered]
    result.signals = final_signals; result.filtered = all_filtered; result.diagnostics = diag
    return result



def apply_cooldown_tiered(signals: list[TradeSignal]) -> list[TradeSignal]:
    now = time.time()
    passed = []
    for sig in signals:
        if is_4h_strategy(sig.strategy_id):
            cooldown = CONFIG_4H.get("cooldown_sec", CONFIG["correlation"]["cooldown_sec"])
        elif is_2h_strategy(sig.strategy_id):
            cooldown = CONFIG_2H.get("cooldown_sec", CONFIG["correlation"]["cooldown_sec"])
        else:
            cooldown = CONFIG["correlation"]["cooldown_sec"]
        key = cooldown_key(sig.pair, sig.strategy_id)
        last = correlation_filter.pair_cooldowns.get(key)
        if last is None:
            # legacy fallback for old entries before tiered cooldown keys
            last = correlation_filter.pair_cooldowns.get(sig.pair)
        if last and now - last < cooldown:
            sig._filtered = True; sig._filter_reason = f"Cooldown: {sig.pair} {now - last:.0f}s/{cooldown}s"; continue
        passed.append(sig)
    return passed


def get_correlation_state() -> dict:
    return correlation_filter.get_state()

def reset_correlation_state():
    correlation_filter.reset()

if __name__ == "__main__":
    print("Multi-Strategy Crypto Futures Engine loaded.")
    print(f"Strategies: {len(ALL_STRATEGIES)}"); print(f"Pairs: {len(CONFIG['pairs'])}"); print(f"Target: ~8.7 signals/hour"); print(f"Max concurrent: {CONFIG['max_concurrent_trades']}"); print(f"Min trade: ${CONFIG['min_trade_size']}")
    for s in ALL_STRATEGIES:
        print(f" {s.id:<20} {s.name:<35} ~{s.avg_signals_per_hour}/hr Lev {s.leverage}x")
