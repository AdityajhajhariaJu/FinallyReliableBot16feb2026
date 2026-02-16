# Hyperliquid Bot — Recreate & Run (one-shot runbook)

This runbook is meant for Spark (OpenClaw assistant) to recreate the Hyperliquid bot setup quickly if `/mnt/botdisk` was detached/wiped.

## 0) Preconditions

1. A detachable volume should be attached and mounted at `/mnt/botdisk` (preferred). If a new disk exists but is not mounted, recreate mount:
   - Format ext4, mount to `/mnt/botdisk`
   - Add `/etc/fstab` with `nofail` so boot survives detaching.

2. Do **NOT** store secrets in this runbook. Credentials go in `.env` only.

## 1) Create project skeleton

Target path: `/mnt/botdisk/bots/hyperliquid`

Create:
- `.venv` (python3 venv)
- `requirements.txt`
- `src/hyperbot/` package
- `README.md`, `.env.example`

Python deps (minimal):
- requests
- websockets
- python-dotenv
- hyperliquid-python-sdk

## 2) Bot components

### Trading bot
Module: `src/hyperbot/bot.py`
Features:
- scans top N coins by day notional from `POST /info {type: metaAndAssetCtxs}`
- EMA crossover signal (fast/slow)
- IOC orders to fill fast (cross spread using current best bid/ask)
- size rounding based on `szDecimals` from universe metadata
- risk controls via env:
  - `HB_TRADE_USD`, `HB_MAX_POS_USD`, `HB_MIN_NOTIONAL`
  - `HB_MAX_LEVERAGE=2`, `HB_IS_CROSS=true`
  - `HB_MIN_SCORE`
- coin auto-ban for 1h on repeated `invalid size/invalid price`
- consults `news_flags.json` for:
  - risk-off pause
  - bias blocks (bullish => block shorts, bearish => block longs)
- writes `startup.flag` on start
- appends structured JSON lines to `trades.log`

### News watcher
Module: `src/hyperbot/news_watcher.py`
- Pull RSS every ~180s from:
  - CoinDesk: https://www.coindesk.com/arc/outboundfeeds/rss/
  - Cointelegraph: https://cointelegraph.com/rss
  - Decrypt: https://decrypt.co/feed
- Writes `news_flags.json` containing:
  - `risk_off_until`
  - `bias` map (coin->dir/until/reason)
  - `last_headlines`

### CLI helpers
- `tail_trades.py`: prints new JSON lines since last offset (for notifier).

## 3) .env configuration (template)

Create `.env` with:

- Hyperliquid:
  - `HL_NETWORK=mainnet|testnet`
  - `HL_ACCOUNT_ADDRESS=0x...` (MAIN wallet address)
  - `HL_PRIVATE_KEY=0x...` (API wallet key preferred)
  - `HL_ARM_TRADING=true|false`

- Bot params (suggested defaults):
  - `HB_INTERVAL_S=2`
  - `HB_FAST_N=12`
  - `HB_SLOW_N=26`
  - `HB_MODE=top`
  - `HB_TOP_N=20`
  - `HB_TRADE_USD=15.0`
  - `HB_MAX_POS_USD=20.0`
  - `HB_MIN_NOTIONAL=10.0`
  - `HB_MIN_SCORE=0.00010`
  - `HB_MAX_LEVERAGE=2`
  - `HB_IS_CROSS=true`
  - `HB_COOLDOWN_S=0`
  - `HB_NEWS_ENABLED=true`
  - `HB_NEWS_RISK_OFF_MIN=30`
  - `HB_NEWS_BIAS_MIN=60`

## 4) Keep running 24×7

### systemd user services
Create + enable:
- `~/.config/systemd/user/hyperbot.service`
  - WorkingDirectory: `/mnt/botdisk/bots/hyperliquid`
  - ExecStart: `.venv/bin/python -m src.hyperbot.bot`
  - Restart=always

- `~/.config/systemd/user/hypernews.service`
  - loop: run news_watcher then sleep 180

Enable lingering:
- `sudo loginctl enable-linger ubuntu`

## 5) OpenClaw cron notifiers (optional but recommended)

Recreate these cron jobs if missing:
- Trade log notifier (every 10s): reads `tail_trades.py` and sends Telegram updates.
- Startup notifier (every 15s): watches `startup.flag`.
- Health guard (every 60s): watches systemd `NRestarts` and restarts service if down.

## 6) One-command recreation

When Aditya says "recreate the Hyperliquid bot", Spark should:
1) ensure `/mnt/botdisk` mounted
2) (re)create project files
3) write `.env.example` and ask user for credentials to fill `.env` (or use provided)
4) enable + start systemd services
5) ensure cron notifiers exist

