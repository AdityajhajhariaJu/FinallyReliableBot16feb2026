#!/usr/bin/env python3
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/opt/multi-strat-engine')
REPORTS = ROOT / 'reports'
OUT_JSON = REPORTS / 'watchdog_summary.json'
OUT_STRAT = REPORTS / 'watchdog_strategy_metrics.csv'
OUT_BLOCK = REPORTS / 'watchdog_blocker_metrics.csv'

TRADE_LOG = REPORTS / 'trade_entry_log.csv'
FILTERED = REPORTS / 'filtered_signals.csv'
HIGH_DROP = REPORTS / 'high_conf_drops.csv'
CYCLE_DIAG = REPORTS / 'cycle_diag_log.csv'


def _read_dict_rows(path: Path):
    if not path.exists():
        return []
    try:
        with path.open() as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                return []
            return list(r)
    except Exception:
        return []


def _to_float(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def _to_int(v, d=0):
    try:
        return int(float(v))
    except Exception:
        return d


def main():
    now = int(time.time())
    rows = _read_dict_rows(TRADE_LOG)

    exec_count = Counter()
    closed_stats = defaultdict(lambda: {"closed": 0, "wins": 0, "losses": 0, "net_pnl": 0.0})

    for r in rows:
        sid = (r.get('strategy_id') or '').strip()
        if not sid:
            continue
        exec_count[sid] += 1
        pnl_s = (r.get('realized_pnl') or '').strip()
        if pnl_s == '':
            continue
        pnl = _to_float(pnl_s, 0.0)
        cs = closed_stats[sid]
        cs['closed'] += 1
        cs['net_pnl'] += pnl
        if pnl > 0:
            cs['wins'] += 1
        elif pnl < 0:
            cs['losses'] += 1

    # Block reasons (merge filtered + high_conf_drops)
    block_count = Counter()
    reason_count = defaultdict(Counter)

    for r in _read_dict_rows(FILTERED):
        sid = (r.get('strategy_id') or '').strip()
        reason = (r.get('reason') or '').strip() or 'unknown'
        if sid:
            block_count[sid] += 1
            reason_count[sid][reason] += 1

    for r in _read_dict_rows(HIGH_DROP):
        sid = (r.get('strategy_id') or '').strip()
        reason = (r.get('reason') or '').strip() or 'unknown'
        if sid:
            block_count[sid] += 1
            reason_count[sid][reason] += 1

    # Cycle suffocation
    diag_rows = _read_dict_rows(CYCLE_DIAG)
    recent_diag = diag_rows[-500:] if len(diag_rows) > 500 else diag_rows
    raw_total = sum(_to_int(x.get('raw')) for x in recent_diag)
    final_total = sum(_to_int(x.get('final')) for x in recent_diag)
    raw_nonzero = sum(1 for x in recent_diag if _to_int(x.get('raw')) > 0)
    raw_zero_final_zero = sum(1 for x in recent_diag if _to_int(x.get('raw')) > 0 and _to_int(x.get('final')) == 0)

    # Concentration (last 100 entries)
    last_100 = rows[:100] if rows else []
    last100_counts = Counter((r.get('strategy_id') or '').strip() for r in last_100 if (r.get('strategy_id') or '').strip())
    top_sid, top_n = ('', 0)
    if last100_counts:
        top_sid, top_n = last100_counts.most_common(1)[0]
    concentration = (top_n / max(1, len(last_100)))

    # Strategy table
    out_rows = []
    alerts = []
    all_sids = sorted(set(exec_count.keys()) | set(block_count.keys()))
    for sid in all_sids:
        e = exec_count.get(sid, 0)
        b = block_count.get(sid, 0)
        cs = closed_stats[sid]
        closed = cs['closed']
        wins = cs['wins']
        net = cs['net_pnl']
        wr = (wins / closed * 100.0) if closed else 0.0
        block_ratio = (b / e) if e else (999.0 if b else 0.0)
        top_reasons = '; '.join(f"{k}:{v}" for k, v in reason_count[sid].most_common(3))

        status = 'WATCH'
        if closed >= 10 and net > 0 and wr >= 45:
            status = 'KEEP'
        if closed >= 10 and net < 0:
            status = 'PAUSE'

        if b >= 10 and block_ratio >= 2.0:
            alerts.append(f"overblocked:{sid} blocks={b} exec={e} ratio={block_ratio:.2f}")

        out_rows.append({
            'strategy_id': sid,
            'executed': e,
            'blocked': b,
            'block_to_exec_ratio': f"{block_ratio:.3f}",
            'closed_trades': closed,
            'win_rate_pct': f"{wr:.2f}",
            'net_pnl': f"{net:.6f}",
            'status': status,
            'top_block_reasons': top_reasons,
        })

    if raw_nonzero >= 20 and raw_zero_final_zero / max(1, raw_nonzero) >= 0.5:
        alerts.append(f"pipeline_suffocation: raw>0 but final=0 in {raw_zero_final_zero}/{raw_nonzero} cycles")

    if len(last_100) >= 20 and concentration >= 0.45:
        alerts.append(f"concentration_risk: {top_sid} share={concentration:.1%} ({top_n}/{len(last_100)})")

    summary = {
        'ts': now,
        'window': {
            'cycle_rows_used': len(recent_diag),
            'trade_rows_used': len(rows),
        },
        'pipeline': {
            'raw_total': raw_total,
            'final_total': final_total,
            'throughput_ratio': (final_total / raw_total) if raw_total else 0.0,
            'raw_nonzero_cycles': raw_nonzero,
            'raw_nonzero_but_final_zero_cycles': raw_zero_final_zero,
        },
        'concentration': {
            'top_strategy': top_sid,
            'top_share_last100': concentration,
            'top_count_last100': top_n,
            'sample_size': len(last_100),
        },
        'alerts': alerts,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))

    with OUT_STRAT.open('w', newline='') as f:
        fields = ['strategy_id', 'executed', 'blocked', 'block_to_exec_ratio', 'closed_trades', 'win_rate_pct', 'net_pnl', 'status', 'top_block_reasons']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(out_rows, key=lambda x: (x['status'], -float(x['net_pnl']))):
            w.writerow(r)

    with OUT_BLOCK.open('w', newline='') as f:
        fields = ['strategy_id', 'reason', 'count']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sid in sorted(reason_count.keys()):
            for reason, cnt in reason_count[sid].most_common():
                w.writerow({'strategy_id': sid, 'reason': reason, 'count': cnt})

    print(OUT_JSON)
    print(OUT_STRAT)
    print(OUT_BLOCK)
    print('alerts', len(alerts))


if __name__ == '__main__':
    main()
