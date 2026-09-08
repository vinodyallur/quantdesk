"""SQLite bar cache.

Historical bars are immutable once a bar closes, so caching them locally makes
backtest iteration fast and keeps you well inside API rate limits. Keyed by
(symbol, timeframe, ts) so different bar sizes coexist.

SQLite is plenty here: a year of 5-minute crypto bars for a dozen symbols is
around a million rows, which it handles without complaint.

Timestamps are stored as signed epoch seconds and read back with
:func:`~quantdesk.core.types.from_epoch` rather than ``datetime.fromtimestamp``, which
rejects negative values on Windows. Long index history starts in 1871, so most of it is
negative and the naive call raises ``OSError`` on exactly the rows this cache exists to
hold.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from quantdesk.core.types import Bar, from_epoch

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol      TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    ts          INTEGER NOT NULL,          -- epoch seconds, UTC
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL    NOT NULL,
    vwap        REAL,
    trade_count INTEGER,
    PRIMARY KEY (symbol, timeframe, ts)
);
CREATE INDEX IF NOT EXISTS idx_bars_lookup ON bars (symbol, timeframe, ts);
"""


class BarCache:
    """Local store of historical bars."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        # WAL keeps reads fast while the live desk is writing.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.commit()

    # ------------------------------------------------------------------ write
    def put(self, timeframe: str, bars: list[Bar]) -> int:
        """Upsert bars. Returns the number of rows written."""
        if not bars:
            return 0
        rows = [
            (
                b.symbol,
                timeframe,
                int(b.ts.timestamp()),
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                b.vwap,
                b.trade_count,
            )
            for b in bars
        ]
        with self._conn:
            self._conn.executemany(
                "INSERT INTO bars (symbol, timeframe, ts, open, high, low, close,"
                " volume, vwap, trade_count) VALUES (?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(symbol, timeframe, ts) DO UPDATE SET"
                " open=excluded.open, high=excluded.high, low=excluded.low,"
                " close=excluded.close, volume=excluded.volume,"
                " vwap=excluded.vwap, trade_count=excluded.trade_count",
                rows,
            )
        return len(rows)

    # ------------------------------------------------------------------- read
    def get(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[Bar]:
        """Read bars ascending by time."""
        sql = "SELECT symbol, ts, open, high, low, close, volume, vwap, trade_count" \
              " FROM bars WHERE symbol=? AND timeframe=?"
        params: list[object] = [symbol, timeframe]
        if start is not None:
            sql += " AND ts >= ?"
            params.append(int(start.timestamp()))
        if end is not None:
            sql += " AND ts <= ?"
            params.append(int(end.timestamp()))
        # To honour `limit` as "most recent N" we sort desc, slice, then reverse.
        sql += " ORDER BY ts DESC" if limit else " ORDER BY ts ASC"
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))

        cur = self._conn.execute(sql, params)
        bars = [
            Bar(
                symbol=r[0],
                ts=from_epoch(r[1]),
                open=r[2],
                high=r[3],
                low=r[4],
                close=r[5],
                volume=r[6],
                vwap=r[7],
                trade_count=r[8],
            )
            for r in cur.fetchall()
        ]
        if limit:
            bars.reverse()
        return bars

    def coverage(self, symbol: str, timeframe: str) -> tuple[datetime, datetime] | None:
        """Earliest and latest cached bar for a symbol, or None if empty."""
        cur = self._conn.execute(
            "SELECT MIN(ts), MAX(ts) FROM bars WHERE symbol=? AND timeframe=?",
            (symbol, timeframe),
        )
        lo, hi = cur.fetchone()
        if lo is None:
            return None
        return from_epoch(lo), from_epoch(hi)

    def count(self, symbol: str | None = None, timeframe: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM bars WHERE 1=1"
        params: list[object] = []
        if symbol:
            sql += " AND symbol=?"
            params.append(symbol)
        if timeframe:
            sql += " AND timeframe=?"
            params.append(timeframe)
        return int(self._conn.execute(sql, params).fetchone()[0])

    def close(self) -> None:
        self._conn.close()
