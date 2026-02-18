#!/usr/bin/env bash
set -euo pipefail
/opt/multi-strat-engine/.venv/bin/python /opt/multi-strat-engine/reports/assistant_tools/suffocation_watchdog.py >/opt/multi-strat-engine/reports/watchdog_run.log 2>&1 || true
