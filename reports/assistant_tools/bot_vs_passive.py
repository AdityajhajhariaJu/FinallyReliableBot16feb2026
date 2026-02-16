#!/usr/bin/env python3
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import ccxt

ROOT = Path('/opt/multi-strat-engine')
REPORTS = ROOT / 'reports'
TRADE_LOG = REPORTS / 'trade_entry_log.csv'
CONFIG_PATH = ROOT / 'config.binance_futures_live.json'

OUT_JSON = REPORTS / 'bot_vs_passive_weekly.json'
OUT_CSV = REPORTS / 'bot_vs_passive_weekly.csv'


@dataclass
class BotStats:
    closed: int
    wins: int
    losses: int
    win_rate_pct: float
    net_pnl: float
    avg_pnl: float


def _to_float(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def load_bot_stats(start_ts: int, end_ts: int) -> BotStats:
    if not TRADE_LOG.exists():
        return BotStats(0, 0, 0, 0.0, 0.0, 0.0)

    vals = []
    with TRADE_LOG.open() as f:
        r = csv.DictReader(f)
        for row in r:
            ets = _to_float(row.get('entry_ts'), 0)
            if ets < start_ts or ets > end_ts:
                continue
            pnl_s = (row.get('realized_pnl') or '').strip()
            if pnl_s == '':
                continue
            vals.append(_to_float(pnl_s, 0.0))

    closed = len(vals)
    wins = sum(1 for v in vals if v > 0)
    losses = sum(1 for v in vals if v < 0)
    wr = (wins / closed * 100.0) if closed else 0.0
    net = sum(vals)
    avg = (net / closed) if closed else 0.0
    return BotStats(closed, wins, losses, wr, net, avg)


def fetch_return_pct(exchange: ccxt.Exchange, symbol: str, start_ts: int, end_ts: int) -> float:
    # Use 1h candles and compare first open vs last close inside range
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', since=start_ts * 1000, limit=500)
    if not ohlcv:
        return 0.0
    filt = [c for c in ohlcv if start_ts * 1000 <= c[0] <= end_ts * 1000]
    if len(filt) < 2:
        filt = ohlcv
    first_open = float(filt[0][1])
    last_close = float(filt[-1][4])
    if first_open <= 0:
        return 0.0
    return (last_close - first_open) / first_open * 100.0


def main():
    now = int(time.time())
    start_ts = now - 7 * 24 * 3600

    cfg = json.loads(CONFIG_PATH.read_text())
    exchange = ccxt.binance({
        'apiKey': cfg.get('exchange', {}).get('key'),
        'secret': cfg.get('exchange', {}).get('secret'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })

    bot = load_bot_stats(start_ts, now)

    btc_ret = fetch_return_pct(exchange, 'BTC/USDT:USDT', start_ts, now)
    eth_ret = fetch_return_pct(exchange, 'ETH/USDT:USDT', start_ts, now)
    mix_ret = (btc_ret + eth_ret) / 2.0

    # Heuristic score (no full equity curve available)
    # Compare signal quality + net realized pnl sign against passive backdrop.
    verdict = 'HOLD'
    notes = []
    if bot.closed < 10:
        verdict = 'LOW_SAMPLE'
        notes.append('Closed trades < 10 this week; comparison has low confidence.')
    else:
        if bot.net_pnl <= 0 and mix_ret > 0:
            verdict = 'UNDERPERFORMING_PASSIVE'
            notes.append('Bot net realized pnl is negative while passive benchmark is positive.')
        elif bot.net_pnl > 0 and bot.win_rate_pct >= 45:
            verdict = 'ACCEPTABLE_ACTIVE'
            notes.append('Bot is profitable this week with acceptable win rate.')

    out = {
        'period': {
            'start_ts': start_ts,
            'end_ts': now,
            'days': 7,
        },
        'bot': {
            'closed_trades': bot.closed,
            'wins': bot.wins,
            'losses': bot.losses,
            'win_rate_pct': round(bot.win_rate_pct, 2),
            'net_realized_pnl': round(bot.net_pnl, 6),
            'avg_pnl_per_trade': round(bot.avg_pnl, 6),
        },
        'passive': {
            'btc_return_pct_7d': round(btc_ret, 4),
            'eth_return_pct_7d': round(eth_ret, 4),
            'btc_eth_50_50_return_pct_7d': round(mix_ret, 4),
        },
        'verdict': verdict,
        'notes': notes,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))

    with OUT_CSV.open('w', newline='') as f:
        fields = [
            'start_ts', 'end_ts', 'bot_closed_trades', 'bot_wins', 'bot_losses', 'bot_win_rate_pct',
            'bot_net_realized_pnl', 'bot_avg_pnl_per_trade', 'btc_return_pct_7d', 'eth_return_pct_7d',
            'btc_eth_50_50_return_pct_7d', 'verdict', 'notes'
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            'start_ts': start_ts,
            'end_ts': now,
            'bot_closed_trades': bot.closed,
            'bot_wins': bot.wins,
            'bot_losses': bot.losses,
            'bot_win_rate_pct': round(bot.win_rate_pct, 2),
            'bot_net_realized_pnl': round(bot.net_pnl, 6),
            'bot_avg_pnl_per_trade': round(bot.avg_pnl, 6),
            'btc_return_pct_7d': round(btc_ret, 4),
            'eth_return_pct_7d': round(eth_ret, 4),
            'btc_eth_50_50_return_pct_7d': round(mix_ret, 4),
            'verdict': verdict,
            'notes': ' | '.join(notes),
        })

    print(OUT_JSON)
    print(OUT_CSV)


if __name__ == '__main__':
    main()
