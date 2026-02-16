#!/usr/bin/env bash
set -euo pipefail

# Recreate Hyperliquid bot project on /opt/olddisk.
# NOTE: This script does NOT write secrets. Fill /opt/olddisk/bots/hyperliquid/.env yourself.

BOTDIR=${BOTDIR:-/opt/olddisk/bots/hyperliquid}
mkdir -p "$BOTDIR"
cd "$BOTDIR"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

cat > requirements.txt <<'REQ'
requests==2.32.3
websockets==12.0
python-dotenv==1.0.1
hyperliquid-python-sdk==0.19.0
REQ

pip install -r requirements.txt

mkdir -p src/hyperbot

cat > .env.example <<'ENV'
HL_NETWORK=mainnet
HL_ACCOUNT_ADDRESS=
HL_PRIVATE_KEY=
HL_ARM_TRADING=false

HB_INTERVAL_S=2
HB_FAST_N=12
HB_SLOW_N=26
HB_MODE=top
HB_TOP_N=20
HB_TRADE_USD=15.0
HB_MAX_POS_USD=20.0
HB_MIN_NOTIONAL=10.0
HB_MIN_SCORE=0.00010
HB_MAX_LEVERAGE=2
HB_IS_CROSS=true
HB_COOLDOWN_S=0

HB_NEWS_ENABLED=true
HB_NEWS_RISK_OFF_MIN=30
HB_NEWS_BIAS_MIN=60
HB_TRADES_LOG=trades.log
ENV

# Copy source files from current workspace snapshot if present (assistant typically writes them directly).
# This script is intentionally minimal; Spark can overwrite bot code via tool-driven writes.

echo "Created venv + requirements + .env.example at $BOTDIR"
echo "Next: create src/hyperbot/*.py and systemd units, then enable services."
