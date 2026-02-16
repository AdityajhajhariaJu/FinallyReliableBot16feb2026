#!/usr/bin/env python3
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path('/opt/multi-strat-engine')
REPORTS = ROOT / 'reports'
TRADE_LOG = REPORTS / 'trade_entry_log.csv'
OUT_JSON = REPORTS / 'regime_pnl_report.json'
OUT_CSV = REPORTS / 'regime_pnl_by_day.csv'


def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def bucket_from_trade(row):
    sid = (row.get('strategy_id') or '').strip()
    conf = _f(row.get('confidence'), 0.0)
    side = (row.get('side') or '').upper()

    # Heuristic market-regime labeling from strategy style + confidence behavior.
    if sid.endswith('_4h') or 'trend' in sid or sid in {'macd_money_map_trend', 'ema50_break_pullback_continuation'}:
        return 'directional'
    if any(k in sid for k in ['reversion', 'keltner', 'cmf_divergence', 'vwap']):
        return 'range'
    if any(k in sid for k in ['breakout', 'squeeze', 'orderblock', 'fvg', 'impulse', 'utbot', 'atr']):
        return 'volatile_structured'
    if conf >= 0.80:
        return 'volatile_structured'
    return 'mixed'


def main():
    if not TRADE_LOG.exists():
        OUT_JSON.write_text(json.dumps({'error': 'missing trade log'}))
        print(OUT_JSON)
        return

    rows = []
    with TRADE_LOG.open() as f:
        r = csv.DictReader(f)
        rows = list(r)

    by_day = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'wins': 0, 'net': 0.0}))
    overall = defaultdict(lambda: {'n': 0, 'wins': 0, 'net': 0.0})

    for row in rows:
        pnl_s = (row.get('realized_pnl') or '').strip()
        if pnl_s == '':
            continue
        pnl = _f(pnl_s)
        day = (row.get('entry_ts_ist') or 'unknown')[:10]
        reg = bucket_from_trade(row)

        d = by_day[day][reg]
        d['n'] += 1
        d['net'] += pnl
        if pnl > 0:
            d['wins'] += 1

        o = overall[reg]
        o['n'] += 1
        o['net'] += pnl
        if pnl > 0:
            o['wins'] += 1

    # Write CSV
    with OUT_CSV.open('w', newline='') as f:
        fields = ['day', 'regime', 'trades', 'wins', 'win_rate_pct', 'net_pnl', 'avg_pnl']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for day in sorted(by_day.keys()):
            for reg, st in by_day[day].items():
                n = st['n']
                wr = (st['wins'] / n * 100.0) if n else 0.0
                avg = (st['net'] / n) if n else 0.0
                w.writerow({
                    'day': day,
                    'regime': reg,
                    'trades': n,
                    'wins': st['wins'],
                    'win_rate_pct': round(wr, 2),
                    'net_pnl': round(st['net'], 6),
                    'avg_pnl': round(avg, 6),
                })

    summary = {
        'ts': int(time.time()),
        'overall_by_regime': {},
        'best_regime_by_net': None,
        'worst_regime_by_net': None,
    }

    ranked = []
    for reg, st in overall.items():
        n = st['n']
        wr = (st['wins'] / n * 100.0) if n else 0.0
        avg = (st['net'] / n) if n else 0.0
        summary['overall_by_regime'][reg] = {
            'trades': n,
            'wins': st['wins'],
            'win_rate_pct': round(wr, 2),
            'net_pnl': round(st['net'], 6),
            'avg_pnl': round(avg, 6),
        }
        ranked.append((st['net'], reg))

    if ranked:
        ranked.sort()
        summary['worst_regime_by_net'] = {'regime': ranked[0][1], 'net_pnl': round(ranked[0][0], 6)}
        summary['best_regime_by_net'] = {'regime': ranked[-1][1], 'net_pnl': round(ranked[-1][0], 6)}

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(OUT_JSON)
    print(OUT_CSV)


if __name__ == '__main__':
    main()
