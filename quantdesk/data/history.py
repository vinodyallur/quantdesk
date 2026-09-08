"""Long-history data sources.

The desk's live feeds cover recent history at intraday resolution. Anything that needs
to see more than one market cycle - regime models, drawdown expectations, whether a
pattern's edge survives a era it was not designed in - needs decades, and no crypto
venue can supply them.

**What is actually available, which is less than "100 years of everything":**

======================  ==========================  ==============  ==========
source                  coverage                    resolution      key needed
======================  ==========================  ==============  ==========
Shiller ``ie_data``     Jan 1871 onward             monthly         no
Yahoo ``^GSPC``         Jan 1928 onward             daily           no
Alpaca US equities      ~2016 onward                intraday        yes
Binance / Bybit perps   ~2019 onward                intraday        no
======================  ==========================  ==============  ==========

So a century is real for equity indices and does not exist for crypto: spot is about
fifteen years old and perpetual futures about nine. Any long-history model has to be
developed on index data and then transferred, and that transfer is an assumption to be
tested rather than a detail. Claiming a century of crypto history would be inventing it.

**On the Shiller series specifically.** Its price column is a *monthly average of daily
closes*, not a month's open-high-low-close. Treating it as OHLC would understate range
and make any stop-based backtest optimistic, so :func:`ShillerSource.fetch` sets all four
prices equal and the richer columns - CAPE, dividends, earnings, CPI, the 10-year yield -
are returned separately by :meth:`ShillerSource.records`. Use it for regime and valuation
work, not for simulating fills.
"""

from __future__ import annotations

import io
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from quantdesk.core.types import Bar, from_epoch

log = logging.getLogger(__name__)

#: A browser user agent. Both Yahoo and the Yale host reject the default urllib agent.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
"""Shiller's own copy. ``shillerdata.com`` hosts the landing page but not this path."""

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"


class HistoryError(RuntimeError):
    """A history source could not supply data. Never raised for merely empty results."""


def _get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise HistoryError(f"{url} returned HTTP {exc.code} ({exc.reason})") from exc
    except OSError as exc:
        raise HistoryError(f"{url} unreachable: {exc}") from exc


# --------------------------------------------------------------------- sources


class HistorySource(ABC):
    """A source of long-run bars for one symbol at a time."""

    #: Timeframe label these bars are stored under, matching ``Settings.timeframe``
    #: vocabulary where it overlaps.
    timeframe: str = "1Day"

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Bar]:
        """Bars ascending by timestamp. Empty list when the range holds nothing."""


class YahooChartSource(HistorySource):
    """Daily index and equity history from Yahoo's chart endpoint.

    The chart endpoint still answers without a cookie or crumb, unlike the ``v7``
    download and quote endpoints which now require both. It is unofficial either way, so
    failures are reported rather than retried indefinitely - a data source that has
    started refusing you is something to know about, not to paper over.

    ``^GSPC`` reaches back to January 1928, which is where the century actually comes
    from. Note that pre-1962 rows carry zero volume: the index existed but the volume
    series does not, and treating those zeros as real would break any volume filter.
    """

    def __init__(self, interval: str = "1d", timeout: float = 60.0) -> None:
        if interval not in ("1d", "1wk", "1mo"):
            raise ValueError("interval must be one of 1d, 1wk, 1mo")
        self.interval = interval
        self.timeout = timeout
        self.timeframe = {"1d": "1Day", "1wk": "1Week", "1mo": "1Month"}[interval]

    def fetch(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Bar]:
        # Defaults span everything the endpoint will give. 1900 is comfortably before
        # any listing, and the endpoint clamps to the real first trade date.
        p1 = int((start or datetime(1900, 1, 1, tzinfo=timezone.utc)).timestamp())
        p2 = int((end or datetime.now(timezone.utc)).timestamp())
        url = (
            f"{YAHOO_CHART_URL}{urllib.parse.quote(symbol, safe='')}"
            f"?period1={p1}&period2={p2}&interval={self.interval}"
        )
        payload = json.loads(_get(url, self.timeout))
        return self._parse(symbol, payload)

    def _parse(self, symbol: str, payload: dict) -> list[Bar]:
        chart = payload.get("chart") or {}
        if chart.get("error"):
            raise HistoryError(f"Yahoo rejected {symbol}: {chart['error']}")
        results = chart.get("result") or []
        if not results:
            return []
        result = results[0]
        stamps = result.get("timestamp") or []
        quotes = (result.get("indicators") or {}).get("quote") or [{}]
        q = quotes[0]
        opens, highs = q.get("open") or [], q.get("high") or []
        lows, closes = q.get("low") or [], q.get("close") or []
        volumes = q.get("volume") or []

        bars: list[Bar] = []
        for i, stamp in enumerate(stamps):
            close = _at(closes, i)
            if close is None:
                # Yahoo emits nulls for holidays it has a timestamp but no print for.
                # Carrying them forward would invent prices; skipping is the honest
                # option and leaves a gap the caller can see.
                continue
            open_ = _at(opens, i)
            high = _at(highs, i)
            low = _at(lows, i)
            bars.append(
                Bar(
                    symbol=symbol,
                    ts=from_epoch(stamp),
                    open=float(open_ if open_ is not None else close),
                    high=float(high if high is not None else close),
                    low=float(low if low is not None else close),
                    close=float(close),
                    volume=float(_at(volumes, i) or 0.0),
                )
            )
        bars.sort(key=lambda b: b.ts)
        return bars


@dataclass(slots=True)
class ShillerRecord:
    """One month of Shiller's series, with the valuation columns kept.

    These are the reason to reach for this source at all: a price series alone is
    available elsewhere with better resolution, but CAPE back to 1881 is not.
    """

    ts: datetime
    price: float
    """Monthly *average* of daily closes, not a month-end print."""
    dividend: float | None = None
    earnings: float | None = None
    cpi: float | None = None
    long_rate: float | None = None
    """10-year Treasury yield, percent."""
    cape: float | None = None
    """Cyclically adjusted P/E. Begins 1881, since it needs ten prior years."""

    @property
    def real_price(self) -> float | None:
        """Price in constant dollars, if CPI is present."""
        return None if not self.cpi else self.price / self.cpi


class ShillerSource(HistorySource):
    """Monthly US equity history from January 1871, via Robert Shiller's dataset.

    Requires ``xlrd``, because the file is published in the pre-2007 ``.xls`` format that
    ``openpyxl`` cannot open. The dependency is imported lazily so that the rest of the
    desk installs and runs without it.

    The published file trails the present by a year or more; it is a research dataset,
    not a feed. Check :meth:`coverage` rather than assuming it reaches today.
    """

    timeframe = "1Month"

    def __init__(self, url: str = SHILLER_URL, timeout: float = 120.0) -> None:
        self.url = url
        self.timeout = timeout
        self._records: list[ShillerRecord] | None = None
        self._raw: bytes | None = None

    def fetch(
        self,
        symbol: str = "SP500",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Bar]:
        """Monthly bars with all four prices equal to the monthly average.

        Equal OHLC is deliberate and is not a placeholder to be filled in later. The
        source has no intramonth range, so inventing one - say by using neighbouring
        months as a proxy high and low - would put fabricated numbers underneath any
        stop-distance calculation.
        """
        out = [
            Bar(
                symbol=symbol,
                ts=r.ts,
                open=r.price,
                high=r.price,
                low=r.price,
                close=r.price,
                volume=0.0,
            )
            for r in self.records()
            if (start is None or r.ts >= start) and (end is None or r.ts <= end)
        ]
        return out

    def records(self, refresh: bool = False) -> list[ShillerRecord]:
        """Parsed monthly rows, downloaded once and held."""
        if self._records is None or refresh:
            self._raw = _get(self.url, self.timeout)
            self._records = self.parse(self._raw)
        return self._records

    def coverage(self) -> tuple[datetime, datetime] | None:
        rows = self.records()
        return (rows[0].ts, rows[-1].ts) if rows else None

    @staticmethod
    def parse(blob: bytes) -> list[ShillerRecord]:
        """Parse the workbook's ``Data`` sheet into records.

        The header sits on row 8 and the first seven rows are prose, so the offset is
        hardcoded. If Shiller republishes with a different preamble this raises rather
        than silently reading titles as data.
        """
        try:
            import xlrd  # noqa: F401  - imported for the clear error, used via pandas
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise HistoryError(
                "Shiller's dataset is published as a legacy .xls file, which needs "
                "xlrd. Install it with: pip install xlrd==2.0.2"
            ) from exc
        import pandas as pd

        book = pd.ExcelFile(io.BytesIO(blob), engine="xlrd")
        if "Data" not in book.sheet_names:
            raise HistoryError(
                f"expected a 'Data' sheet, found {book.sheet_names}. The file layout "
                f"has changed and the parser needs updating."
            )
        frame = book.parse("Data", header=7)
        if "Date" not in frame.columns:
            raise HistoryError(
                "no 'Date' column on the Data sheet; the header offset has moved"
            )

        # Addressed by label rather than position. ``itertuples`` would rename
        # "Rate GS10" to a positional ``_6`` because of the space, which silently points
        # at a different column the moment Shiller inserts one.
        def column(label: str) -> list:
            return list(frame[label]) if label in frame.columns else []

        dates = column("Date")
        prices = column("P")
        dividends = column("D")
        earnings = column("E")
        cpis = column("CPI")
        rates = column("Rate GS10")
        capes = column("CAPE")

        records: list[ShillerRecord] = []
        for i in range(len(dates)):
            ts = _shiller_date(_at(dates, i))
            price = _number(_at(prices, i))
            if ts is None or price is None:
                # Trailing notes and blank separator rows. Skipping them is right, but
                # only after the date fails to parse - never by truncating at a guess.
                continue
            records.append(
                ShillerRecord(
                    ts=ts,
                    price=price,
                    dividend=_number(_at(dividends, i)),
                    earnings=_number(_at(earnings, i)),
                    cpi=_number(_at(cpis, i)),
                    long_rate=_number(_at(rates, i)),
                    cape=_number(_at(capes, i)),
                )
            )
        records.sort(key=lambda r: r.ts)
        return records


# ------------------------------------------------------------------ resampling


def resample_monthly(bars: list[Bar], allow_partial: bool = False) -> list[Bar]:
    """Aggregate finer bars into calendar months.

    Two properties matter more than the aggregation itself:

    **No lookahead.** Each output bar is built only from bars inside its own month, and
    is stamped with the first constituent's timestamp so it sits at the period start like
    every other bar in the system.

    **The unfinished month is dropped.** A month still in progress is not a bar that
    existed at the time, and including it lets a backtest act on a close that had not
    happened. ``allow_partial`` exists for live display, where the incomplete month is
    what you want to look at, and should stay ``False`` for anything being measured.
    """
    if not bars:
        return []
    ordered = sorted(bars, key=lambda b: b.ts)
    groups: dict[tuple[str, int, int], list[Bar]] = {}
    order: list[tuple[str, int, int]] = []
    for bar in ordered:
        key = (bar.symbol, bar.ts.year, bar.ts.month)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(bar)

    if not allow_partial:
        # The final month per symbol is only complete once a later month has begun.
        latest: dict[str, tuple[int, int]] = {}
        for symbol, year, month in order:
            latest[symbol] = max(latest.get(symbol, (0, 0)), (year, month))
        order = [k for k in order if (k[1], k[2]) != latest[k[0]]]

    out: list[Bar] = []
    for key in order:
        members = groups[key]
        out.append(
            Bar(
                symbol=key[0],
                ts=members[0].ts,
                open=members[0].open,
                high=max(b.high for b in members),
                low=min(b.low for b in members),
                close=members[-1].close,
                volume=sum(b.volume for b in members),
                trade_count=(
                    sum(b.trade_count or 0 for b in members)
                    if any(b.trade_count is not None for b in members)
                    else None
                ),
            )
        )
    return out


# ----------------------------------------------------------------------- store


class HistoryStore:
    """Cache-first access to long history.

    The point is that a century of daily bars is downloaded once. The Yahoo endpoint is
    unofficial and rate limited, and the Shiller workbook is 1.6 MB of Excel; neither
    should be hit on every backtest.
    """

    def __init__(self, cache) -> None:
        self.cache = cache

    def load(
        self,
        source: HistorySource,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        refresh: bool = False,
    ) -> list[Bar]:
        """Return bars, fetching and caching only when needed."""
        if not refresh:
            cached = self.cache.get(symbol, source.timeframe, start, end)
            if cached:
                log.info(
                    "history: %d cached %s bars for %s", len(cached), source.timeframe, symbol
                )
                return cached

        bars = source.fetch(symbol, start, end)
        if bars:
            written = self.cache.put(source.timeframe, bars)
            log.info(
                "history: fetched %d %s bars for %s from %s, cached %d",
                len(bars), source.timeframe, symbol, source.name, written,
            )
        return bars


# ------------------------------------------------------------------- internals


def _at(seq: list, index: int):
    return seq[index] if index < len(seq) else None


def _number(value) -> float | None:
    """Coerce a spreadsheet cell to a float, or None if it is not one."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    # NaN is how pandas represents an empty cell, and it is not a number we can use.
    return None if out != out else out


def _shiller_date(value) -> datetime | None:
    """Convert Shiller's ``YYYY.MM`` decimal to a month start.

    The encoding is a trap: October is written ``.1`` and January ``.01``, and as a float
    those are 0.1 and 0.01. Multiplying the fraction by 100 and rounding recovers both,
    whereas reading the digits after the decimal point as text turns October into
    January.
    """
    number = _number(value)
    if number is None:
        return None
    year = int(number)
    if not 1800 <= year <= 2200:
        return None
    month = int(round((number - year) * 100))
    if month == 0:
        month = 1
    if not 1 <= month <= 12:
        return None
    return datetime(year, month, 1, tzinfo=timezone.utc)
