"""
Strategy Activation Policy (Market-Regime Aware)

SECTION 1) Criteria for ACTIVATING strategies
- Volatility bands (ATR%):
  - Low-vol floor: atr_pct >= 0.0008 (0.08%)
  - Extreme-vol cap: atr_pct <= 0.0600 (6.00%)
- One-sided / trend market indicators:
  - one_sided=True when directional move is persistent over lookback
  - trend_strength from EMA(20)-EMA(50) gap on 15m
- Volume engagement:
  - volume_ratio = latest_vol / avg_vol_20
  - minimum engagement >= 0.35 (global)
- Strategy category activation:
  - trend: one_sided OR trend_strength >= 0.0025, volume_ratio >= 0.8
  - breakout: volatility_expanding=True AND volume_ratio >= 1.05
  - reversion: NOT one_sided, moderate volatility, volume_ratio >= 0.7
  - scalp: non-extreme volatility, volume_ratio >= 0.6

SECTION 2) Criteria for DEACTIVATING strategies
- Low market engagement: volume_ratio < 0.35
- Dead market / too flat: atr_pct < 0.0008
- Extreme chaotic market: atr_pct > 0.0600
- Category-specific deactivation:
  - trend disabled in flat/choppy regimes (weak trend)
  - reversion disabled in strong one-sided trend
  - breakout disabled when volatility contraction + weak volume

SECTION 3) Strategy list and intended conditions
- trend:
  - mtf_ema_ribbon, ema_cross_rsi, macd_hist_flip, adx_di_cross
  - best in one-sided / directional conditions
- breakout:
  - bb_squeeze, bb_kc_squeeze, atr_breakout, donchian_breakout
  - best in volatility expansion + strong participation
- reversion:
  - rsi_snapback, keltner_reversion, cmf_divergence, vwap_bounce
  - best in balanced/choppy markets with mean-reversion behavior
- event/momentum:
  - liquidation_cascade
  - best in impulse spikes with elevated activity

SECTION 4) Profit vs loss summary (expected impact)
- Expected profit impact:
  - Better alignment of strategy to market regime can reduce low-quality entries.
  - Fewer false breakouts in dead markets and fewer mean-reversion fades in one-way trends.
- Expected loss impact:
  - Reduced overtrading in poor conditions lowers repeated small losses/fees.
- Risk note:
  - Thresholds are intentionally moderate (not too strict) to avoid killing signal flow.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import os
import time
import csv
from pathlib import Path


Category = Literal["trend", "breakout", "reversion", "scalp", "event", "unknown"]

ACTIVATION_POLICY_DEBUG = os.getenv("ACTIVATION_POLICY_DEBUG", "1") == "1"
ACTIVATION_POLICY_DEBUG_PATH = Path("/opt/multi-strat-engine/reports/activation_policy_debug.csv")
YT_SCALP_LITE_MODE = os.getenv("YT_SCALP_LITE_MODE", "0") == "1"


@dataclass
class MarketRegime:
    atr_pct: float
    atr_pct_15m: float
    volume_ratio: float
    trend_strength: float
    one_sided: bool
    volatility_expanding: bool


def _ema(values: list[float], period: int):
    if len(values) < period:
        return None
    alpha = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = alpha * v + (1 - alpha) * e
    return e


def _atr_pct(candles, period=14):
    if not candles or len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h = candles[i].high
        l = candles[i].low
        pc = candles[i - 1].close
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    atr = sum(trs[-period:]) / period if len(trs) >= period else 0.0
    px = candles[-1].close or 1.0
    return max(0.0, atr / px)


def _volume_ratio(candles, lookback=20):
    if not candles:
        return 0.0
    if len(candles) < lookback + 1:
        return 1.0
    latest = candles[-1].volume
    avg = sum(c.volume for c in candles[-lookback - 1:-1]) / lookback
    if avg <= 0:
        return 1.0
    return latest / avg


def _one_sided(candles_15m, lookback=12, min_dir_ratio=0.67):
    if not candles_15m or len(candles_15m) < lookback + 1:
        return False
    closes = [c.close for c in candles_15m[-(lookback + 1):]]
    ups = 0
    dns = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            ups += 1
        elif closes[i] < closes[i - 1]:
            dns += 1
    total = max(1, ups + dns)
    return (ups / total) >= min_dir_ratio or (dns / total) >= min_dir_ratio


def _trend_strength(candles_15m):
    if not candles_15m or len(candles_15m) < 60:
        return 0.0
    closes = [c.close for c in candles_15m]
    e20 = _ema(closes[-80:], 20)
    e50 = _ema(closes[-120:], 50)
    px = closes[-1] or 1.0
    if e20 is None or e50 is None:
        return 0.0
    return abs(e20 - e50) / px


def detect_market_regime(candles_1m, candles_5m=None, candles_15m=None) -> MarketRegime:
    candles_5m = candles_5m or candles_1m
    candles_15m = candles_15m or candles_5m

    atr1 = _atr_pct(candles_1m, 14)
    atr15 = _atr_pct(candles_15m, 14)
    vol = _volume_ratio(candles_1m, 20)
    tr = _trend_strength(candles_15m)
    one_way = _one_sided(candles_15m, lookback=12, min_dir_ratio=0.67)

    # Expansion when short-term vol exceeds medium-term baseline meaningfully
    vol_exp = atr1 > (atr15 * 1.10) if atr15 > 0 else atr1 > 0.002

    return MarketRegime(
        atr_pct=atr1,
        atr_pct_15m=atr15,
        volume_ratio=vol,
        trend_strength=tr,
        one_sided=one_way,
        volatility_expanding=vol_exp,
    )


STRATEGY_CATEGORY_MAP: dict[str, Category] = {
    # trend
    "mtf_ema_ribbon": "trend",
    "ema_cross_rsi": "trend",
    "macd_hist_flip": "trend",
    "adx_di_cross": "trend",

    # breakout
    "bb_squeeze": "breakout",
    "bb_kc_squeeze": "breakout",
    "atr_breakout": "breakout",
    "donchian_breakout": "breakout",

    # reversion
    "rsi_snapback": "reversion",
    "keltner_reversion": "reversion",
    "cmf_divergence": "reversion",
    "vwap_bounce": "reversion",

    # event / momentum
    "liquidation_cascade": "event",
}


def infer_category(strategy_id: str, strategy_category: str = "") -> Category:
    sid = (strategy_id or "").strip().lower()
    if sid in STRATEGY_CATEGORY_MAP:
        return STRATEGY_CATEGORY_MAP[sid]

    sc = (strategy_category or "").strip().lower()
    if sc in ("trend", "reversion", "breakout"):
        return sc  # type: ignore[return-value]

    # Heuristic fallback (safe, moderate)
    if "ema" in sid or "adx" in sid or "trend" in sid:
        return "trend"
    if "squeeze" in sid or "breakout" in sid:
        return "breakout"
    if "rsi" in sid or "reversion" in sid or "vwap" in sid or "cmf" in sid:
        return "reversion"
    if "liq" in sid or "cascade" in sid:
        return "event"
    return "unknown"


YT_SUPER_STRONG = {
    "ema_cci_macd_combo",
    "ema_ribbon_33889_pullback",
    "ema10_20_cci_momentum",
    "price_action_dle",
    "topdown_aoi_shift",
    "impulse_macd_regime_breakout",
    "ema50_break_pullback_continuation",
    "orderblock_mtf_inducement_breaker",
    "fvg_bos_inversion_mtf",
    "ema20_zone_heikin_rsi",
    "fib_precision_respect_met",
    "ema_rsi_stoch_tripwire",
    "utbot_atr_adaptive_trend",
    "ma_slope_crossover_sr",
    "vsa_volume_truth",
}

# User-prioritized winners: apply relaxed regime gating so they fire more often.
RELAXED_REGIME_STRATEGIES = {
    "structure_break_ob_4h",
    "ema20_zone_heikin_rsi",
    "macd_money_map_trend",
    "orderblock_mtf_inducement_breaker",
}


def _yt_regime_bucket(r: MarketRegime) -> str:
    # Coarse, stable buckets for routing YT strategies only
    if r.atr_pct < 0.0010 or (not r.one_sided and r.trend_strength < 0.0016 and r.volume_ratio < 0.95):
        return "choppy"
    if r.volatility_expanding and r.atr_pct >= 0.0014:
        return "expansion"
    if r.one_sided and r.trend_strength >= 0.0028:
        return "strong_trend"
    if r.trend_strength >= 0.0018:
        return "transition"
    return "structured"


def _debug_policy(pair: str, strategy_id: str, strategy_category: str, ok: bool, reason: str, regime: MarketRegime):
    if not ACTIVATION_POLICY_DEBUG:
        return
    try:
        ACTIVATION_POLICY_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
        exists = ACTIVATION_POLICY_DEBUG_PATH.exists()
        with ACTIVATION_POLICY_DEBUG_PATH.open("a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["ts", "pair", "strategy_id", "category", "ok", "reason", "atr_pct", "atr_pct_15m", "volume_ratio", "trend_strength", "one_sided", "vol_expanding"])
            w.writerow([
                int(time.time()), pair or "", strategy_id or "", strategy_category or "", int(bool(ok)), reason,
                f"{regime.atr_pct:.6f}", f"{regime.atr_pct_15m:.6f}", f"{regime.volume_ratio:.4f}",
                f"{regime.trend_strength:.6f}", int(bool(regime.one_sided)), int(bool(regime.volatility_expanding))
            ])
    except Exception:
        pass


def should_activate_strategy(strategy_id: str, strategy_category: str, regime: MarketRegime, pair: str = ""):
    cat = infer_category(strategy_id, strategy_category)
    sid = (strategy_id or "").strip().lower()

    # Relaxed gating for user-prioritized winners.
    if sid in RELAXED_REGIME_STRATEGIES:
        if regime.atr_pct > 0.0800:
            why = f"relaxed_block_extreme_volatility(atr={regime.atr_pct:.4f})"
            _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
            return False, why
        if regime.atr_pct < 0.0005:
            why = f"relaxed_block_dead_market(atr={regime.atr_pct:.4f})"
            _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
            return False, why
        if regime.volume_ratio < 0.20:
            why = f"relaxed_block_low_engagement(vol_ratio={regime.volume_ratio:.2f})"
            _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
            return False, why
        why = "relaxed_regime_ok"
        _debug_policy(pair, strategy_id, strategy_category, True, why, regime)
        return True, why

    # Global deactivation guards (kept moderate)
    if regime.volume_ratio < 0.35:
        why = f"low_engagement(vol_ratio={regime.volume_ratio:.2f})"
        _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
        return False, why
    if regime.atr_pct < 0.0008:
        why = f"dead_market(atr={regime.atr_pct:.4f})"
        _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
        return False, why
    if regime.atr_pct > 0.0600:
        why = f"extreme_volatility(atr={regime.atr_pct:.4f})"
        _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
        return False, why

    # User-requested: regime shortlist applies to NEW/YT strategies only.
    if sid in YT_SUPER_STRONG:
        # Optional lite mode for faster/scalping YT systems.
        yt_scalp_lite = {
            "ema_cci_macd_combo", "ema10_20_cci_momentum", "ema20_zone_heikin_rsi",
            "ema_rsi_stoch_tripwire", "utbot_atr_adaptive_trend",
        }
        if YT_SCALP_LITE_MODE and sid in yt_scalp_lite:
            if regime.atr_pct < 0.0007:
                why = f"yt_lite_block_dead_market(atr={regime.atr_pct:.4f})"
                _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
                return False, why
            why = "yt_lite_ok"
            _debug_policy(pair, strategy_id, strategy_category, True, why, regime)
            return True, why

        bucket = _yt_regime_bucket(regime)
        allowed = {
            "strong_trend": {
                "ema_ribbon_33889_pullback", "ema50_break_pullback_continuation", "ema_rsi_stoch_tripwire",
                "ema20_zone_heikin_rsi", "utbot_atr_adaptive_trend",
            },
            "transition": {
                "ema10_20_cci_momentum", "ma_slope_crossover_sr", "ema_cci_macd_combo", "utbot_atr_adaptive_trend",
            },
            "structured": {
                "topdown_aoi_shift", "price_action_dle", "fib_precision_respect_met", "orderblock_mtf_inducement_breaker", "vsa_volume_truth",
            },
            "expansion": {
                "impulse_macd_regime_breakout", "fvg_bos_inversion_mtf", "utbot_atr_adaptive_trend",
            },
            "choppy": {
                "fib_precision_respect_met", "orderblock_mtf_inducement_breaker", "vsa_volume_truth",
            },
        }
        if sid in allowed.get(bucket, set()):
            why = f"yt_regime_ok:{bucket}"
            _debug_policy(pair, strategy_id, strategy_category, True, why, regime)
            return True, why
        why = f"yt_regime_block:{bucket}"
        _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
        return False, why

    if cat == "trend":
        if regime.one_sided or regime.trend_strength >= 0.0025:
            if regime.volume_ratio >= 0.8:
                why = "trend_ok"
                _debug_policy(pair, strategy_id, strategy_category, True, why, regime)
                return True, why
            why = f"trend_low_volume(vol_ratio={regime.volume_ratio:.2f})"
            _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
            return False, why
        why = f"trend_not_directional(trend={regime.trend_strength:.4f})"
        _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
        return False, why

    if cat == "breakout":
        if regime.volatility_expanding and regime.volume_ratio >= 1.05:
            why = "breakout_ok"
            _debug_policy(pair, strategy_id, strategy_category, True, why, regime)
            return True, why
        why = f"breakout_no_expansion(expand={regime.volatility_expanding},vol={regime.volume_ratio:.2f})"
        _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
        return False, why

    if cat == "reversion":
        if regime.one_sided and regime.trend_strength >= 0.0035:
            why = "reversion_blocked_one_sided"
            _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
            return False, why
        if regime.volume_ratio < 0.7:
            why = f"reversion_low_volume(vol_ratio={regime.volume_ratio:.2f})"
            _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
            return False, why
        if regime.atr_pct > 0.0300:
            why = f"reversion_too_volatile(atr={regime.atr_pct:.4f})"
            _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
            return False, why
        why = "reversion_ok"
        _debug_policy(pair, strategy_id, strategy_category, True, why, regime)
        return True, why

    if cat == "event":
        if regime.volume_ratio >= 1.0 and regime.atr_pct >= 0.0015:
            why = "event_ok"
            _debug_policy(pair, strategy_id, strategy_category, True, why, regime)
            return True, why
        why = f"event_insufficient_impulse(vol={regime.volume_ratio:.2f},atr={regime.atr_pct:.4f})"
        _debug_policy(pair, strategy_id, strategy_category, False, why, regime)
        return False, why

    # unknown/scalp fallback: permissive, only block extremes
    why = "fallback_ok"
    _debug_policy(pair, strategy_id, strategy_category, True, why, regime)
    return True, why
