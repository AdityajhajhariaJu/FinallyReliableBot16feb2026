#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path

ENTRY_LOG = Path('/opt/multi-strat-engine/reports/trade_entry_log.csv')
OUT_JSON = Path('/opt/multi-strat-engine/reports/strategy_interference_matrix.json')
OUT_CSV = Path('/opt/multi-strat-engine/reports/strategy_interference_pairs.csv')
WINDOW_SEC = 90 * 60  # 90 minutes


def rf(x, d=6):
    return round(float(x), d)


def load_closed_rows():
    rows = []
    with ENTRY_LOG.open() as fh:
        r = csv.DictReader(fh)
        for row in r:
            sid = (row.get('strategy_id') or '').strip()
            pair = (row.get('pair') or '').strip()
            ts_s = (row.get('entry_ts') or '').strip()
            pnl_s = (row.get('realized_pnl') or '').strip()
            if not sid or not pair or not ts_s or not pnl_s:
                continue
            try:
                ts = int(float(ts_s))
                pnl = float(pnl_s)
            except Exception:
                continue
            rows.append({'strategy_id': sid, 'pair': pair, 'entry_ts': ts, 'pnl': pnl})
    rows.sort(key=lambda x: x['entry_ts'])
    return rows


def main():
    if not ENTRY_LOG.exists():
        raise SystemExit('trade_entry_log.csv not found')

    rows = load_closed_rows()
    if not rows:
        OUT_JSON.write_text(json.dumps({'closed_trades_used': 0, 'pairs': []}, indent=2))
        OUT_CSV.write_text('base_strategy,partner_strategy,samples,base_avg_pnl_solo,base_avg_pnl_with_partner,pnl_lift,base_winrate_solo,base_winrate_with_partner,winrate_lift\n')
        print(OUT_JSON)
        print(OUT_CSV)
        return

    # index rows by pair for local window checks
    by_pair = defaultdict(list)
    for i, row in enumerate(rows):
        by_pair[row['pair']].append((i, row))

    solo_stats = defaultdict(lambda: {'n': 0, 'pnl': 0.0, 'wins': 0})
    with_partner = defaultdict(lambda: {'n': 0, 'pnl': 0.0, 'wins': 0})  # key=(base,partner)

    for pair, items in by_pair.items():
        n = len(items)
        for idx, row in items:
            base = row['strategy_id']
            ts = row['entry_ts']
            pnl = row['pnl']

            partners = set()
            # local scan over same pair rows
            for jdx, other in items:
                if jdx == idx:
                    continue
                dt = abs(other['entry_ts'] - ts)
                if dt <= WINDOW_SEC:
                    partners.add(other['strategy_id'])

            if not partners:
                solo_stats[base]['n'] += 1
                solo_stats[base]['pnl'] += pnl
                solo_stats[base]['wins'] += 1 if pnl > 0 else 0
            else:
                for p in partners:
                    k = (base, p)
                    with_partner[k]['n'] += 1
                    with_partner[k]['pnl'] += pnl
                    with_partner[k]['wins'] += 1 if pnl > 0 else 0

    out_rows = []
    for (base, partner), d in with_partner.items():
        n = d['n']
        if n < 3:
            continue
        sp = solo_stats[base]
        nsolo = sp['n']
        base_avg_solo = (sp['pnl'] / nsolo) if nsolo else 0.0
        base_wr_solo = (sp['wins'] / nsolo) if nsolo else 0.0

        with_avg = d['pnl'] / n if n else 0.0
        with_wr = d['wins'] / n if n else 0.0

        out_rows.append({
            'base_strategy': base,
            'partner_strategy': partner,
            'samples': n,
            'base_avg_pnl_solo': rf(base_avg_solo, 6),
            'base_avg_pnl_with_partner': rf(with_avg, 6),
            'pnl_lift': rf(with_avg - base_avg_solo, 6),
            'base_winrate_solo': rf(base_wr_solo, 4),
            'base_winrate_with_partner': rf(with_wr, 4),
            'winrate_lift': rf(with_wr - base_wr_solo, 4),
        })

    out_rows.sort(key=lambda x: (x['pnl_lift'], x['winrate_lift']), reverse=True)

    OUT_JSON.write_text(json.dumps({'closed_trades_used': len(rows), 'window_sec': WINDOW_SEC, 'pairs': out_rows}, indent=2))
    with OUT_CSV.open('w', newline='') as fh:
        fields = ['base_strategy', 'partner_strategy', 'samples', 'base_avg_pnl_solo', 'base_avg_pnl_with_partner', 'pnl_lift', 'base_winrate_solo', 'base_winrate_with_partner', 'winrate_lift']
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(OUT_JSON)
    print(OUT_CSV)


if __name__ == '__main__':
    main()
