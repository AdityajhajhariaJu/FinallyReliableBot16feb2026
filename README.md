# FinallyReliableBot16feb2026

A live multi-strategy Binance USDT-M futures trading engine with 1m/2h/4h strategy packs, regime-aware strategy activation, robust risk filters, execution safeguards, and analytics/report tooling.

This repository is prepared for deployment/share without secrets or runtime data.

---

## Highlights

- Multi-timeframe strategy engine
  - 1m execution strategies
  - 2h pack (mostly 5m/15m logic with longer hold profile)
  - 4h pack for higher-timeframe conviction
- Regime-aware activation/deactivation
  - policy-driven strategy gating by volatility/trend/volume context
- Execution protections
  - cooldown controls
  - same-side flood limits
  - category caps
  - per-pair and global trade caps
  - TP/SL placement and repair flow
- Diagnostics and observability
  - cycle funnel (`raw -> cooldown -> flood -> category -> final`)
  - filtered/high-confidence drop logging
  - strategy interference, robustness, and calibration tools
  - SQLite trade logger support (`trade_logger.py`)
- Daily health analytics
  - suffocation watchdog
  - regime-vs-PnL report
  - bot-vs-passive benchmark report

---

## Repository structure

### Core runtime
- `trade_loop.py` — main live loop (market fetch, scan, execute, reconcile)
- `strategies.py` — strategy definitions + scan pipeline + filters + risk controls
- `new_strategies_2h.py` — 2h strategy pack
- `new_strategies_4h.py` — 4h strategy pack
- `strategy_activation_policy.py` — regime activation logic
- `news_bias.py` — optional confidence bias from news
- `trade_logger.py` — SQLite lifecycle logger

### Reports & analysis
- `reports/*.py` — diagnostics dashboards and helpers
- `reports/assistant_tools/*.py` — deeper analytics scripts:
  - confidence calibration
  - robustness checks
  - strategy interference matrix
  - watchdogs and comparison reports

### Operational docs/helpers
- `runbooks/` — operational runbooks
- `scripts/` — helper scripts

---

## Setup

### 1) Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Config
Copy and fill:
- `config.binance_futures_live.example.json` -> `config.binance_futures_live.json`

> Never commit live keys/secrets.

### 3) Run
```bash
python3 trade_loop.py
```

---

## Key runtime behavior

### Signal pipeline
1. Fetch market data
2. Evaluate strategies
3. Apply policy and quality gates
4. Apply cooldown/flood/category/slot controls
5. Execute selected trades
6. Track and report performance

### Cooldown model
- Pair cooldowns by strategy tier (1m/2h/4h separated)
- Post-close cooldown support
- Strategy-level cooldown overrides for specific strategies when configured

---

## Monitoring outputs

Common outputs under `reports/`:
- `trade_entry_log.csv`
- `trade_exit_log.csv`
- `cycle_diag_log.csv`
- `filtered_signals.csv`
- `high_conf_drops.csv`
- `watchdog_summary.json`
- `regime_pnl_report.json`
- `bot_vs_passive_weekly.json`

---

## Safety and repo hygiene

Excluded from git:
- live API keys/configs
- `.env` files
- runtime CSV/DB artifacts
- logs/cache files

See `.gitignore` for full patterns.

---

## Recommended workflow

1. Make changes in small increments
2. Compile-check Python files
3. Restart service and verify logs
4. Validate with post-change metrics (watchdog + funnel + PnL)
5. Commit with clear message

---

## Disclaimer

Live derivatives trading is high risk and may result in significant losses. Use at your own discretion and with proper risk controls.
