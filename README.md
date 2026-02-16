# FinallyReliableBot16feb2026

A practical, live **multi-strategy Binance USDT-M futures bot** designed to run continuously, manage risk, and produce traceable performance analytics.

This README is written for both:
- people who can read code, and
- people who just want to understand what the bot does and how to run it safely.

---

## 1) What this bot does (in simple words)

The bot scans crypto futures markets, looks for trade setups from multiple strategy styles, filters weak setups, places orders with TP/SL, and logs everything for analysis.

Think of it as a pipeline:
1. **Find opportunities** (many strategies across multiple timeframes)
2. **Reject bad/unsafe signals** (cooldowns, category caps, confidence checks)
3. **Execute trades** (market entry + reduce-only TP/SL)
4. **Track outcomes** (PnL logs, diagnostics, watchdog reports)
5. **Continuously compare performance** (strategy-level + regime-level + passive benchmark)

---

## 2) Timeframe model

The bot has three practical layers:

- **1m layer (fast/tactical)**
  - more frequent opportunities
  - more noise-sensitive
- **2h pack (medium horizon, often using 5m/15m internals)**
  - slower than 1m
  - usually more selective
- **4h pack (higher conviction)**
  - fewer trades
  - generally more stable setups

---

## 3) Strategy families (human explanation)

This bot includes many strategies, but they mostly fall into these buckets:

### A) Trend strategies
Try to follow directional moves.
Examples: EMA-based trend systems, momentum continuation.

### B) Reversion strategies
Assume price may revert toward mean/structure after an extreme move.
Examples: Keltner reversion, CMF divergence variants.

### C) Structural / smart-money-style strategies
Focus on zones/levels/price behavior patterns.
Examples: order blocks, FVG-style concepts, structure breaks.

### D) Volatility / breakout strategies
Look for expansion from compression.
Examples: squeeze/breakout style logic.

### E) Volume-informed strategies
Use volume as conviction signal (effort vs result concept).
Examples: `vsa_volume_truth`, `funding_fade` style contextual filters.

> Note: Strategy list evolves. See `strategies.py`, `new_strategies_2h.py`, and `new_strategies_4h.py` for exact live IDs and parameters.

---

## 4) Core architecture (important files)

### Runtime engine
- `trade_loop.py`
  - main live loop
  - data fetch, execution, TP/SL maintenance, reconciliation
- `strategies.py`
  - strategy registry
  - signal generation pipeline
  - trade sizing, leverage capping, and core filters
- `new_strategies_2h.py`
  - 2h pack strategy definitions/export
- `new_strategies_4h.py`
  - 4h strategy definitions/export
- `strategy_activation_policy.py`
  - regime-aware activation/deactivation policy
- `trade_logger.py`
  - SQLite logger for signal/trade lifecycle and performance stats
- `news_bias.py`
  - optional confidence biasing from external news context

### Analytics + diagnostics
- `reports/assistant_tools/suffocation_watchdog.py`
  - detects over-filtering and concentration risk
- `reports/assistant_tools/regime_pnl_report.py`
  - maps PnL by market regime buckets
- `reports/assistant_tools/bot_vs_passive.py`
  - compares bot performance vs BTC/ETH passive benchmark
- `reports/*.py`
  - additional health checks, blocker dashboards, calibration, etc.

---

## 5) Risk controls in this bot

The bot is not "always trade". It uses protective gates:

- **Confidence threshold checks**
- **Confirmation checks**
- **Cooldowns**
  - signal cooldown
  - post-close cooldown
  - optional strategy-level cooldown overrides
- **Same-side flood control** (avoid piling too many same-direction trades)
- **Category caps** (avoid over-concentration by strategy type)
- **Per-pair max trades**
- **Global max concurrent trades**
- **Leverage caps** (including tier-specific rules)

These controls are why the bot can survive longer, but over-tight settings can reduce opportunity.

---

## 6) Logging and transparency

This project is designed to be auditable.

Typical runtime outputs in `reports/`:
- `trade_entry_log.csv` — entries + strategy IDs + confidence
- `trade_exit_log.csv` — closed trades and realized outcomes
- `cycle_diag_log.csv` — pipeline funnel counts (`raw -> final`)
- `filtered_signals.csv` — what got filtered and why
- `high_conf_drops.csv` — high-confidence signals that still didn’t execute
- `watchdog_summary.json` — health/concentration alerts
- `regime_pnl_report.json` — regime-level profitability
- `bot_vs_passive_weekly.json` — active vs passive comparison

---

## 7) Setup and run

### Prerequisites
- Linux server (recommended)
- Python 3.10+
- Binance Futures API key/secret

### Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Config
Use template:
- `config.binance_futures_live.example.json`

Create live file:
- `config.binance_futures_live.json`

### Start
```bash
python3 trade_loop.py
```

---

## 8) Operational best practices

1. Change one thing at a time
2. Restart service, then verify logs immediately
3. Compare before/after in:
   - cycle diagnostics
   - strategy-level PnL
   - watchdog alerts
4. Avoid overreacting to tiny sample sizes
5. Evaluate strategy changes on meaningful trade counts

---

## 9) Security and repo hygiene

This repository intentionally excludes:
- live API keys
- `.env` secrets
- runtime CSV/DB/log artifacts

See `.gitignore` for full rules.

Never commit:
- `config.binance_futures_live.json`
- private keys or exchange credentials

---

## 10) FAQ

### Q: Why can profitable strategy windows still lose later?
Because market regimes change. A strategy can be strong in volatility expansion and weak in chop.

### Q: Why does the bot sometimes show few/no trades?
Usually due to low-quality market conditions, strict filters, or cooldown overlap.

### Q: Does more strategies always mean more profit?
No. Too many overlapping strategies can increase interference and noise.

### Q: What should I trust most for decision making?
Use rolling evidence from:
- realized PnL
- profit factor
- drawdown behavior
- concentration/suffocation alerts

---

## 11) Disclaimer

This software is for research/automation use. Live derivatives trading is high risk and can lead to significant losses. Use proper risk controls and only risk capital you can afford to lose.
