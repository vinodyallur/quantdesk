"""Perpetual futures: funding rates and open interest.

Funding is the mechanism that keeps a perpetual contract tethered to spot. Longs pay
shorts when the contract trades above the index and shorts pay longs when it trades
below, every eight hours on most venues. That makes the funding rate a direct, published
measure of which side is crowded - information that simply does not exist in spot data,
and the main reason to look at perps at all.

The reading that matters: **persistently positive funding means longs are paying to stay
long**, which is a crowded book rather than a bullish signal. It is a carry cost on every
long position and a carry income on every short, and at 0.01% per eight hours - the
typical baseline - it compounds to roughly 11% a year. Any perp strategy that ignores it
is mis-stating its own returns.

Open interest adds the other half. Rising price on rising open interest is new money
taking the trend; rising price on *falling* open interest is short covering, which
exhausts itself. Murphy makes the same argument for futures volume and open interest in
chapter 7, and it transfers directly.

**Coverage is short, and no amount of engineering fixes that.** Binance's BTCUSDT perp
begins September 2019. Open interest history is worse: the venue serves roughly the last
month only, so a long open-interest series has to be accumulated going forward rather
than backfilled. Both are recorded here rather than hidden, because a model fitted on
seven years of one asset class is a different claim from one fitted on a century.

Both venues serve this data without an API key.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from quantdesk.core.types import from_epoch

log = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

#: Most venues settle funding three times a day. Used to annualise a rate.
FUNDINGS_PER_DAY = 3


class PerpsError(RuntimeError):
    """A perps endpoint could not be read."""


def _get_json(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise PerpsError(f"{url} returned HTTP {exc.code} ({exc.reason})") from exc
    except OSError as exc:
        raise PerpsError(f"{url} unreachable: {exc}") from exc


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


# ------------------------------------------------------------------- records


@dataclass(slots=True)
class FundingRate:
    """One settled funding payment."""

    symbol: str
    ts: datetime
    rate: float
    """Fraction of notional paid by longs to shorts. Negative means shorts pay longs."""
    mark_price: float | None = None

    @property
    def annualised(self) -> float:
        """Rate extrapolated to a year, as a fraction.

        Simple rather than compounded, because it is used to compare a carry cost against
        an expected return over a holding period, not to project a terminal value.
        """
        return self.rate * FUNDINGS_PER_DAY * 365.0

    @property
    def longs_pay(self) -> bool:
        return self.rate > 0


@dataclass(slots=True)
class OpenInterest:
    """Total contracts outstanding at a point in time."""

    symbol: str
    ts: datetime
    contracts: float
    notional: float | None = None
    """Value of open interest in quote currency, where the venue reports it."""


@dataclass
class FundingSummary:
    """What a stretch of funding history says about positioning."""

    symbol: str
    count: int
    mean_rate: float
    median_rate: float
    positive_share: float
    """Fraction of settlements where longs paid. Above ~0.7 is a persistently long book."""
    annualised_cost: float
    """Cost of holding a long across the whole window, as a fraction of notional."""
    start: datetime | None = None
    end: datetime | None = None

    @property
    def crowded_long(self) -> bool:
        """Longs paying most of the time and paying materially.

        Both conditions are required. A book can be long most of the time at a trivial
        rate, which costs nothing and says little; and a single large spike can move the
        mean without indicating any persistent lean.
        """
        return self.positive_share >= 0.70 and self.annualised_cost >= 0.05

    @property
    def crowded_short(self) -> bool:
        return self.positive_share <= 0.30 and self.annualised_cost <= -0.05

    def summary(self) -> list[str]:
        lean = (
            "crowded long"
            if self.crowded_long
            else "crowded short"
            if self.crowded_short
            else "no persistent lean"
        )
        window = ""
        if self.start and self.end:
            window = f"  {self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}"
        return [
            f"{self.symbol}  {self.count} settlements{window}",
            f"mean {self.mean_rate * 100:+.4f}% per funding  "
            f"median {self.median_rate * 100:+.4f}%",
            f"longs paid {self.positive_share:.0%} of the time",
            f"annualised carry {self.annualised_cost:+.2%} for a long  ->  {lean}",
        ]


# ------------------------------------------------------------------- venues


class PerpsSource(ABC):
    """A venue serving perpetual futures funding and open interest."""

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def funding(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        max_records: int = 10_000,
    ) -> list[FundingRate]:
        """Settled funding rates ascending by time."""

    @abstractmethod
    def open_interest(
        self, symbol: str, period: str = "1d", limit: int = 500
    ) -> list[OpenInterest]:
        """Open interest history ascending by time, as far back as the venue keeps it."""


class BinancePerps(PerpsSource):
    """Binance USD-margined futures.

    ``fundingRate`` ignores ``startTime=0`` and answers with the most recent page
    regardless, so reaching the beginning of history means walking forward in explicit
    windows. That behaviour is the reason this class pages rather than making one call.
    """

    BASE = "https://fapi.binance.com"
    PAGE = 1000

    def __init__(self, timeout: float = 30.0, pause: float = 0.2) -> None:
        self.timeout = timeout
        #: Courtesy delay between pages. The endpoint is generous but not unlimited, and
        #: a full history walk is hundreds of calls.
        self.pause = pause

    def funding(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        max_records: int = 10_000,
    ) -> list[FundingRate]:
        sym = _binance_symbol(symbol)
        # Perps did not exist before 2019; starting earlier just wastes empty pages.
        cursor = _ms(start or datetime(2019, 1, 1, tzinfo=timezone.utc))
        ceiling = _ms(end or datetime.now(timezone.utc))
        out: list[FundingRate] = []
        seen: set[int] = set()

        while cursor < ceiling and len(out) < max_records:
            url = (
                f"{self.BASE}/fapi/v1/fundingRate?symbol={urllib.parse.quote(sym)}"
                f"&startTime={cursor}&endTime={ceiling}&limit={self.PAGE}"
            )
            page = _get_json(url, self.timeout)
            if not isinstance(page, list) or not page:
                break
            for row in page:
                stamp = int(row["fundingTime"])
                if stamp in seen:
                    continue
                seen.add(stamp)
                out.append(
                    FundingRate(
                        symbol=symbol,
                        ts=from_epoch(stamp / 1000.0),
                        rate=float(row["fundingRate"]),
                        mark_price=_float_or_none(row.get("markPrice")),
                    )
                )
            newest = max(int(r["fundingTime"]) for r in page)
            if newest <= cursor:
                # No forward progress. Without this the loop spins on a single page.
                break
            cursor = newest + 1
            if len(page) < self.PAGE:
                break
            time.sleep(self.pause)

        out.sort(key=lambda f: f.ts)
        return out[:max_records]

    def open_interest(
        self, symbol: str, period: str = "1d", limit: int = 500
    ) -> list[OpenInterest]:
        """Open interest history.

        Binance keeps roughly the last 30 days of this series regardless of the limit
        asked for, so this is a starting point for accumulation, not a backfill.
        """
        sym = _binance_symbol(symbol)
        url = (
            f"{self.BASE}/futures/data/openInterestHist"
            f"?symbol={urllib.parse.quote(sym)}&period={period}"
            f"&limit={min(500, max(1, limit))}"
        )
        rows = _get_json(url, self.timeout)
        if not isinstance(rows, list):
            return []
        out = [
            OpenInterest(
                symbol=symbol,
                ts=from_epoch(int(r["timestamp"]) / 1000.0),
                contracts=float(r["sumOpenInterest"]),
                notional=_float_or_none(r.get("sumOpenInterestValue")),
            )
            for r in rows
        ]
        out.sort(key=lambda o: o.ts)
        return out


class BybitPerps(PerpsSource):
    """Bybit linear perpetuals.

    Wraps every payload in ``result.list`` and returns newest first, which is the opposite
    of Binance. Worth a second source: when two venues disagree about funding the
    difference is itself tradeable, and it is a check on a single venue's outage.
    """

    BASE = "https://api.bybit.com"
    PAGE = 200

    def __init__(self, timeout: float = 30.0, pause: float = 0.2) -> None:
        self.timeout = timeout
        self.pause = pause

    def _result(self, url: str) -> dict:
        payload = _get_json(url, self.timeout)
        if payload.get("retCode") not in (0, None):
            raise PerpsError(f"Bybit error {payload.get('retCode')}: {payload.get('retMsg')}")
        return payload.get("result") or {}

    def funding(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        max_records: int = 10_000,
    ) -> list[FundingRate]:
        sym = _binance_symbol(symbol)
        floor = _ms(start or datetime(2019, 1, 1, tzinfo=timezone.utc))
        cursor = _ms(end or datetime.now(timezone.utc))
        out: list[FundingRate] = []
        seen: set[int] = set()

        # Bybit pages backwards from endTime, so the walk runs newest to oldest and the
        # result is sorted at the end.
        while cursor > floor and len(out) < max_records:
            url = (
                f"{self.BASE}/v5/market/funding/history?category=linear"
                f"&symbol={urllib.parse.quote(sym)}&startTime={floor}"
                f"&endTime={cursor}&limit={self.PAGE}"
            )
            rows = self._result(url).get("list") or []
            if not rows:
                break
            for row in rows:
                stamp = int(row["fundingRateTimestamp"])
                if stamp in seen:
                    continue
                seen.add(stamp)
                out.append(
                    FundingRate(
                        symbol=symbol,
                        ts=from_epoch(stamp / 1000.0),
                        rate=float(row["fundingRate"]),
                    )
                )
            oldest = min(int(r["fundingRateTimestamp"]) for r in rows)
            if oldest >= cursor:
                break
            cursor = oldest - 1
            if len(rows) < self.PAGE:
                break
            time.sleep(self.pause)

        out.sort(key=lambda f: f.ts)
        return out[:max_records]

    def open_interest(
        self, symbol: str, period: str = "1d", limit: int = 200
    ) -> list[OpenInterest]:
        sym = _binance_symbol(symbol)
        url = (
            f"{self.BASE}/v5/market/open-interest?category=linear"
            f"&symbol={urllib.parse.quote(sym)}&intervalTime={period}"
            f"&limit={min(200, max(1, limit))}"
        )
        rows = self._result(url).get("list") or []
        out = [
            OpenInterest(
                symbol=symbol,
                ts=from_epoch(int(r["timestamp"]) / 1000.0),
                contracts=float(r["openInterest"]),
            )
            for r in rows
        ]
        out.sort(key=lambda o: o.ts)
        return out


# ------------------------------------------------------------------ analysis


def summarise_funding(rates: list[FundingRate], symbol: str = "") -> FundingSummary:
    """Reduce funding history to a positioning read."""
    if not rates:
        return FundingSummary(
            symbol=symbol, count=0, mean_rate=0.0, median_rate=0.0,
            positive_share=0.0, annualised_cost=0.0,
        )
    ordered = sorted(rates, key=lambda f: f.ts)
    values = sorted(f.rate for f in ordered)
    n = len(values)
    mean = sum(values) / n
    mid = n // 2
    median = values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2.0
    positive = sum(1 for v in values if v > 0) / n
    return FundingSummary(
        symbol=symbol or ordered[0].symbol,
        count=n,
        mean_rate=mean,
        median_rate=median,
        positive_share=positive,
        annualised_cost=mean * FUNDINGS_PER_DAY * 365.0,
        start=ordered[0].ts,
        end=ordered[-1].ts,
    )


def open_interest_signal(
    prices: list[float], interest: list[float]
) -> tuple[str, str]:
    """Read price and open interest together, per Murphy's chapter 7 argument.

    Returns ``(reading, explanation)``. Direction alone is not the point: the same price
    move means different things depending on whether positions are being opened or
    closed, and open interest is what distinguishes them.
    """
    if len(prices) < 2 or len(interest) < 2:
        return "unknown", "not enough history to compare"
    price_up = prices[-1] > prices[0]
    oi_up = interest[-1] > interest[0]
    if price_up and oi_up:
        return "new money long", "price and open interest both rising: fresh buying"
    if price_up and not oi_up:
        return "short covering", (
            "price rising while open interest falls: positions are being closed, "
            "which exhausts itself"
        )
    if not price_up and oi_up:
        return "new money short", "price falling on rising open interest: fresh selling"
    return "long liquidation", (
        "price and open interest both falling: longs are giving up, and the selling "
        "runs out when they are done"
    )


# ------------------------------------------------------------------ internals


def _binance_symbol(symbol: str) -> str:
    """Map a desk symbol such as ``BTC/USD`` onto a perp symbol such as ``BTCUSDT``.

    Perps are quoted in USDT, not USD, so a bare concatenation produces a symbol neither
    venue lists. Anything already in venue form is passed through untouched.
    """
    if "/" not in symbol:
        return symbol.upper()
    base, _, quote = symbol.partition("/")
    quote = quote.upper()
    if quote == "USD":
        quote = "USDT"
    return f"{base.upper()}{quote}"


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def default_source(venue: str = "binance") -> PerpsSource:
    venues = {"binance": BinancePerps, "bybit": BybitPerps}
    if venue not in venues:
        raise ValueError(f"unknown venue {venue!r}; choose from {sorted(venues)}")
    return venues[venue]()


__all__ = [
    "BinancePerps",
    "BybitPerps",
    "FundingRate",
    "FundingSummary",
    "OpenInterest",
    "PerpsError",
    "PerpsSource",
    "default_source",
    "open_interest_signal",
    "summarise_funding",
]
