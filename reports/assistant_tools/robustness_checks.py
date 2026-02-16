#!/usr/bin/env python3
"""
Robustness checks for live strategy outcomes.
Implements 4 practical validation layers:
1) Cost/slippage/funding stress
2) Regime-split validation
3) Capacity/liquidity sanity checks
4) Stability checks (rolling + bootstrap)

Input: /opt/multi-strat-engine/reports/trade_exit_log.csv
Output:
- /opt/multi-strat-engine/reports/robustness_checks_summary.json
- /opt/multi-strat-engine/reports/robustness_checks_by_strategy.csv
"""

from __future__ import annotations
import csv
import json
import random
import math
from pathlib import Path
from collections import defaultdict

EXIT_LOG = Path('/opt/multi-strat-engine/reports/trade_exit_log.csv')
OUT_JSON = Path('/opt/multi-strat-engine/reports/robustness_checks_summary.json')
OUT_CSV = Path('/opt/multi-strat-engine/reports/robustness_checks_by_strategy.csv')


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def i(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def parse_rows():
    if not EXIT_LOG.exists():
        return []
    rows = list(csv.DictReader(EXIT_LOG.open()))
    clean = []
    for r in rows:
        pnl_raw = (r.get('realized_pnl') or '').strip()
        # keep blanks in raw dataset, but only numeric pnl for robustness math
        pnl = None
        if pnl_raw != '':
            try:
                pnl = float(pnl_raw)
            except Exception:
                pnl = None
        entry = f(r.get('entry_price'))
        qty = f(r.get('qty'))
        notional = abs(entry * qty)
        ts = i(r.get('exit_ts') or r.get('entry_ts'))
        note = (r.get('note') or '').lower()
        atr_pct = None
        if 'atr%=' in note:
            try:
                seg = note.split('atr%=')[1].split('%')[0]
                atr_pct = float(seg) / 100.0
            except Exception:
                atr_pct = None
        clean.append({
            'trade_id': r.get('trade_id') or '',
            'pair': r.get('pair') or '',
            'side': r.get('side') or '',
            'strategy_id': r.get('strategy_id') or 'unknown',
            'strategy_name': r.get('strategy_name') or '',
            'strategy_tier': r.get('strategy_tier') or '',
            'entry_price': entry,
            'qty': qty,
            'notional': notional,
            'exit_ts': ts,
            'pnl': pnl,
            'atr_pct': atr_pct,
            'note': note,
        })
    return clean


def regime_bucket(row):
    # Prefer parsed ATR%; fallback to tier-based coarse split
    atr = row.get('atr_pct')
    if atr is not None:
        if atr < 0.002:
            return 'low_vol'
        if atr < 0.006:
            return 'mid_vol'
        return 'high_vol'
    tier = (row.get('strategy_tier') or '').lower()
    if tier == '4h':
        return 'slow_regime'
    if tier == '2h':
        return 'swing_regime'
    return 'fast_regime'


def summarize_pnls(vals):
    if not vals:
        return {'n': 0, 'sum': 0.0, 'avg': 0.0, 'win_rate': 0.0, 'std': 0.0}
    n = len(vals)
    s = sum(vals)
    avg = s / n
    wr = sum(1 for v in vals if v > 0) / n
    var = sum((v - avg) ** 2 for v in vals) / n
    return {'n': n, 'sum': round(s, 6), 'avg': round(avg, 6), 'win_rate': round(wr, 4), 'std': round(math.sqrt(var), 6)}


def bootstrap_ci(vals, samples=800):
    if len(vals) < 5:
        return None
    means = []
    n = len(vals)
    for _ in range(samples):
        draw = [vals[random.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    lo = means[int(0.05 * len(means))]
    hi = means[int(0.95 * len(means))]
    return [round(lo, 6), round(hi, 6)]


def main():
    rows = parse_rows()
    numeric = [r for r in rows if r['pnl'] is not None]

    # 1) Cost/slippage/funding stress (synthetic penalties on realized pnl)
    # mild/base/harsh penalties in quote currency terms per trade
    stress = {
        'mild': 0.08,
        'base': 0.20,
        'harsh': 0.40,
    }
    stress_out = {}
    raw_pnls = [r['pnl'] for r in numeric]
    stress_out['raw'] = summarize_pnls(raw_pnls)
    for k, penalty in stress.items():
        adj = [v - penalty for v in raw_pnls]
        stress_out[k] = summarize_pnls(adj)

    # 2) Regime split validation
    by_regime = defaultdict(list)
    for r in numeric:
        by_regime[regime_bucket(r)].append(r['pnl'])
    regime_out = {k: summarize_pnls(v) for k, v in by_regime.items()}

    # 3) Capacity/liquidity sanity checks
    notionals = [r['notional'] for r in numeric if r['notional'] > 0]
    cap = {}
    if notionals:
        n_sorted = sorted(notionals)
        cap = {
            'min_notional': round(n_sorted[0], 4),
            'p50_notional': round(n_sorted[len(n_sorted)//2], 4),
            'p90_notional': round(n_sorted[int(len(n_sorted)*0.9)-1], 4),
            'max_notional': round(n_sorted[-1], 4),
            'tiny_notional_trades_lt5': sum(1 for x in notionals if x < 5),
        }

    # 4) Stability checks
    # rolling windows by close time order
    ordered = sorted(numeric, key=lambda r: r['exit_ts'])
    pnls_ord = [r['pnl'] for r in ordered]
    window = 10
    rolling_means = []
    if len(pnls_ord) >= window:
        for j in range(window, len(pnls_ord)+1):
            segment = pnls_ord[j-window:j]
            rolling_means.append(sum(segment)/window)
    stability = {
        'overall': summarize_pnls(pnls_ord),
        'bootstrap_mean_ci_90': bootstrap_ci(pnls_ord),
        'rolling10_mean_min': round(min(rolling_means), 6) if rolling_means else None,
        'rolling10_mean_max': round(max(rolling_means), 6) if rolling_means else None,
    }

    # by strategy table
    by_strat = defaultdict(list)
    for r in numeric:
        by_strat[r['strategy_id']].append(r['pnl'])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['strategy_id', 'trades', 'sum_pnl', 'avg_pnl', 'win_rate', 'std'])
        for sid, vals in sorted(by_strat.items(), key=lambda kv: sum(kv[1]), reverse=True):
            s = summarize_pnls(vals)
            w.writerow([sid, s['n'], s['sum'], s['avg'], s['win_rate'], s['std']])

    summary = {
        'rows_total': len(rows),
        'rows_with_numeric_pnl': len(numeric),
        'rows_blank_pnl': len(rows) - len(numeric),
        'cost_slippage_funding_stress': stress_out,
        'regime_split': regime_out,
        'capacity_liquidity_sanity': cap,
        'stability': stability,
        'by_strategy_csv': str(OUT_CSV),
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(str(OUT_JSON))
    print(str(OUT_CSV))


if __name__ == '__main__':
    main()
