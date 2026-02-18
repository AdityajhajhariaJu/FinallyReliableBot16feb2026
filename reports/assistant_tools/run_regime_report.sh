#!/usr/bin/env bash
set -euo pipefail
/opt/multi-strat-engine/.venv/bin/python /opt/multi-strat-engine/reports/assistant_tools/regime_pnl_report.py >/opt/multi-strat-engine/reports/regime_pnl_run.log 2>&1 || true
