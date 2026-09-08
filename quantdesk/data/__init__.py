"""Market data: feeds, local cache, long history, and rolling in-memory history."""

from quantdesk.data.cache import BarCache
from quantdesk.data.feed import DataFeed
from quantdesk.data.history import (
    HistoryError,
    HistorySource,
    HistoryStore,
    ShillerSource,
    YahooChartSource,
    resample_monthly,
)
from quantdesk.data.series import BarSeries, MarketBook

__all__ = [
    "BarCache",
    "BarSeries",
    "DataFeed",
    "HistoryError",
    "HistorySource",
    "HistoryStore",
    "MarketBook",
    "ShillerSource",
    "YahooChartSource",
    "build_feed",
    "resample_monthly",
]


def build_feed(settings=None, offline: bool = False) -> DataFeed:
    """Construct the appropriate feed for the current configuration.

    Kept as a function rather than importing both feeds eagerly so an offline run
    never needs the Alpaca SDK to import cleanly.
    """
    from quantdesk.config import get_settings

    settings = settings or get_settings()

    if offline:
        from quantdesk.data.synthetic_feed import SyntheticFeed

        return SyntheticFeed(settings.universe, settings.timeframe)

    from quantdesk.data.alpaca_feed import AlpacaFeed

    return AlpacaFeed(
        symbols=settings.universe,
        timeframe=settings.timeframe,
        asset_class=settings.asset_class,
        api_key=settings.key(),
        secret_key=settings.secret(),
        stock_feed=settings.stock_feed,
        poll_seconds=settings.poll_seconds,
    )
