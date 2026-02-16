"""trade_logger.py — Per-strategy trade performance logger (SQLite).
Production-hardened for async bot usage (thread lock + safe upserts).
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategyStats:
    strategy_id: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    total_fees: float
    net_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    avg_hold_sec: float
    avg_leverage: float
    profit_factor: float
    fee_impact_pct: float
    last_trade_ts: float


class TradeLogger:
    def __init__(self, db_path: str = "/opt/multi-strat-engine/reports/trades.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._create_tables()

    def _exec(self, sql: str, params=(), commit: bool = True):
        with self._lock:
            cur = self.conn.execute(sql, params)
            if commit:
                self.conn.commit()
            return cur

    def _create_tables(self):
        with self._lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    pair TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    confidence REAL,
                    entry_price REAL,
                    tp_price REAL,
                    sl_price REAL,
                    leverage INTEGER,
                    was_executed INTEGER DEFAULT 0,
                    filter_reason TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    pair TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    leverage INTEGER DEFAULT 1,
                    size REAL DEFAULT 0,
                    pnl REAL DEFAULT 0,
                    fees REAL DEFAULT 0,
                    net_pnl REAL DEFAULT 0,
                    entry_ts REAL NOT NULL,
                    exit_ts REAL,
                    hold_seconds REAL DEFAULT 0,
                    exit_reason TEXT DEFAULT '',
                    is_open INTEGER DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades(pair);
                CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(is_open);
                CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_id);
                """
            )
            self.conn.commit()

    def log_signal(self, pair: str, strategy_id: str, side: str,
                   confidence: float | None, entry_price: float | None,
                   tp_price: float | None, sl_price: float | None,
                   leverage: int | None, executed: bool = False,
                   filter_reason: str = ""):
        self._exec(
            """INSERT INTO signals
               (timestamp, pair, strategy_id, side, confidence, entry_price, tp_price, sl_price, leverage, was_executed, filter_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), pair, strategy_id, side, confidence, entry_price, tp_price, sl_price,
             leverage, 1 if executed else 0, filter_reason),
        )

    def log_entry(self, trade_id: str, pair: str, strategy_id: str, side: str,
                  entry_price: float, leverage: int, size: float):
        now = time.time()
        self._exec(
            """
            INSERT INTO trades (trade_id, pair, strategy_id, side, entry_price, leverage, size, entry_ts, is_open)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(trade_id) DO UPDATE SET
                pair=excluded.pair,
                strategy_id=excluded.strategy_id,
                side=excluded.side,
                entry_price=excluded.entry_price,
                leverage=excluded.leverage,
                size=excluded.size,
                entry_ts=excluded.entry_ts,
                is_open=1
            """,
            (trade_id, pair, strategy_id, side, entry_price, leverage, size, now),
        )

    def log_exit(self, trade_id: str, exit_price: float | None, pnl: float | None,
                 fees: float | None = 0.0, reason: str = ""):
        now = time.time()
        with self._lock:
            row = self.conn.execute("SELECT entry_ts FROM trades WHERE trade_id = ?", (trade_id,)).fetchone()
            entry_ts = row[0] if row else now
            hold_sec = max(0.0, now - float(entry_ts))
            pnl_v = float(pnl or 0.0)
            fees_v = float(fees or 0.0)
            net = pnl_v - fees_v
            self.conn.execute(
                """UPDATE trades
                   SET exit_price=?, pnl=?, fees=?, net_pnl=?, exit_ts=?, hold_seconds=?, exit_reason=?, is_open=0
                   WHERE trade_id=?""",
                (exit_price, pnl_v, fees_v, net, now, hold_sec, reason, trade_id),
            )
            self.conn.commit()

    def get_strategy_stats(self, strategy_id: str) -> Optional[StrategyStats]:
        rows = self._exec(
            "SELECT pnl, fees, net_pnl, hold_seconds, leverage FROM trades WHERE strategy_id=? AND is_open=0",
            (strategy_id,),
            commit=False,
        ).fetchall()
        if not rows:
            return None
        total = len(rows)
        wins = [r for r in rows if float(r[2] or 0) > 0]
        losses = [r for r in rows if float(r[2] or 0) <= 0]

        total_pnl = sum(float(r[0] or 0) for r in rows)
        total_fees = sum(float(r[1] or 0) for r in rows)
        net_pnl = sum(float(r[2] or 0) for r in rows)
        avg_hold = sum(float(r[3] or 0) for r in rows) / total
        avg_lev = sum(float(r[4] or 0) for r in rows) / total

        gross_wins = sum(float(r[2] or 0) for r in wins)
        gross_losses = abs(sum(float(r[2] or 0) for r in losses))

        last_ts = self._exec(
            "SELECT MAX(exit_ts) FROM trades WHERE strategy_id=? AND is_open=0",
            (strategy_id,),
            commit=False,
        ).fetchone()[0] or 0

        fee_base = abs(total_pnl) if abs(total_pnl) > 1e-9 else max(abs(net_pnl), 1e-9)

        return StrategyStats(
            strategy_id=strategy_id,
            total_trades=total,
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / total,
            total_pnl=total_pnl,
            total_fees=total_fees,
            net_pnl=net_pnl,
            avg_pnl=net_pnl / total,
            avg_win=gross_wins / len(wins) if wins else 0.0,
            avg_loss=-(gross_losses / len(losses)) if losses else 0.0,
            avg_hold_sec=avg_hold,
            avg_leverage=avg_lev,
            profit_factor=(gross_wins / gross_losses) if gross_losses > 0 else float("inf"),
            fee_impact_pct=(total_fees / fee_base * 100.0),
            last_trade_ts=float(last_ts),
        )

    def close(self):
        with self._lock:
            self.conn.close()
