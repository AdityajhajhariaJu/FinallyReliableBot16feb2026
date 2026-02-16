#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path

ENTRY_LOG = Path('/opt/multi-strat-engine/reports/trade_entry_log.csv')
OUT_JSON = Path('/opt/multi-strat-engine/reports/confidence_calibration_report.json')
OUT_CSV = Path('/opt/multi-strat-engine/reports/confidence_calibration_by_strategy.csv')


def f(x, d=6):
    return round(float(x), d)


def main():
    if not ENTRY_LOG.exists():
        raise SystemExit('trade_entry_log.csv not found')

    rows = []
    with ENTRY_LOG.open() as fh:
        r = csv.DictReader(fh)
        for row in r:
            pnl_s = (row.get('realized_pnl') or '').strip()
            conf_s = (row.get('confidence') or '').strip()
            sid = (row.get('strategy_id') or '').strip()
            if not pnl_s or not conf_s or not sid:
                continue
            try:
                pnl = float(pnl_s)
                conf = float(conf_s)
            except Exception:
                continue
            rows.append((sid, conf, pnl))

    # global bins
    bins = defaultdict(lambda: {'n': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0})
    by_strategy = defaultdict(list)

    for sid, conf, pnl in rows:
        b = min(19, max(0, int(conf / 0.05)))
        bins[b]['n'] += 1
        bins[b]['wins'] += 1 if pnl > 0 else 0
        bins[b]['losses'] += 1 if pnl < 0 else 0
        bins[b]['pnl'] += pnl
        by_strategy[sid].append((conf, pnl))

    bin_rows = []
    for b in sorted(bins):
        lo = b * 0.05
        hi = lo + 0.05
        d = bins[b]
        n = d['n']
        winrate = (d['wins'] / n) if n else 0.0
        bin_rows.append({
            'bin': f'{lo:.2f}-{hi:.2f}',
            'n': n,
            'wins': d['wins'],
            'losses': d['losses'],
            'winrate': f(winrate, 4),
            'net_pnl': f(d['pnl'], 6),
            'avg_pnl': f(d['pnl'] / n, 6) if n else 0.0,
        })

    strat_rows = []
    for sid, vals in by_strategy.items():
        n = len(vals)
        wins = sum(1 for _, p in vals if p > 0)
        losses = sum(1 for _, p in vals if p < 0)
        net = sum(p for _, p in vals)
        avg_conf = sum(c for c, _ in vals) / n
        strat_rows.append({
            'strategy_id': sid,
            'closed_trades': n,
            'avg_confidence': f(avg_conf, 4),
            'winrate': f(wins / n if n else 0.0, 4),
            'net_pnl': f(net, 6),
            'avg_pnl': f(net / n if n else 0.0, 6),
            'wins': wins,
            'losses': losses,
        })

    strat_rows.sort(key=lambda x: x['net_pnl'], reverse=True)

    OUT_JSON.write_text(json.dumps({
        'closed_trades_used': len(rows),
        'confidence_bins': bin_rows,
        'by_strategy': strat_rows,
    }, indent=2))

    with OUT_CSV.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(strat_rows[0].keys()) if strat_rows else ['strategy_id'])
        w.writeheader()
        for r in strat_rows:
            w.writerow(r)

    print(OUT_JSON)
    print(OUT_CSV)


if __name__ == '__main__':
    main()
