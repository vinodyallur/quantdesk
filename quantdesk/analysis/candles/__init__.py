"""Japanese candlesticks, Murphy chapter 12.

Organised by session count the way his reference table is: :mod:`single`,
:mod:`pairs`, :mod:`triples`, with :mod:`geometry` holding the anatomy and
:class:`CandleReader` tying it together.

Usage::

    reader = CandleReader("BTC/USD")
    for bar in bars:
        trend = Bias.from_sign(dow_state.direction.sign)
        for signal in reader.update(bar, trend):
            print(signal.describe())

The trend argument is required rather than optional, because a candle pattern read
without knowing what preceded it is not a candle pattern.
"""

from quantdesk.analysis.candles.geometry import (
    Candle,
    CandleKind,
    CandleSignal,
    SizeContext,
)
from quantdesk.analysis.candles.pairs import pair_strength, read_pairs
from quantdesk.analysis.candles.reader import CandleReader
from quantdesk.analysis.candles.single import read_single, single_strength
from quantdesk.analysis.candles.triples import read_triples, triple_strength

__all__ = [
    "Candle",
    "CandleKind",
    "CandleReader",
    "CandleSignal",
    "SizeContext",
    "pair_strength",
    "read_pairs",
    "read_single",
    "read_triples",
    "single_strength",
    "triple_strength",
]
