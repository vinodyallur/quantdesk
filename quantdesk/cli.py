"""Command line entry point.

Four commands, each corresponding to something you actually want to do:

    quantdesk backtest              replay history through the pipeline
    quantdesk run                   live paper session with the terminal UI
    quantdesk scan                  one-shot analysis, no trading, no UI
    quantdesk doctor                check what is wired up and what is missing

``run`` defaults to the synthetic offline feed. Reaching the network and placing
orders both require explicit flags, so nothing surprising happens on a bare
checkout - and market data works with no API key at all, which is why ``scan`` and
``backtest`` are useful before any credentials exist.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

from quantdesk.config import get_settings


def _setup_logging(level: str, quiet: bool = False) -> None:
    # The terminal UI owns the screen, so logs go to a file rather than over it.
    handlers: list[logging.Handler] = []
    if quiet:
        settings = get_settings()
        handlers.append(logging.FileHandler(settings.data_dir / "quantdesk.log"))
    else:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


# --------------------------------------------------------------------- backtest


def cmd_backtest(args: argparse.Namespace) -> int:
    from quantdesk.backtest import Backtester
    from quantdesk.data import build_feed

    settings = get_settings()
    symbols = args.symbols or settings.universe
    feed = build_feed(settings, offline=args.offline)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    print(f"fetching {args.days}d of {settings.timeframe} bars for {', '.join(symbols)}...")
    history = feed.history(start=start, end=end, symbols=symbols)

    total = sum(len(v) for v in history.values())
    if total == 0:
        print("no bars returned. try --offline for the synthetic feed.", file=sys.stderr)
        return 1
    print(f"got {total} bars across {len(history)} symbols\n")

    result = Backtester(settings).run(history)
    print("=" * 68)
    for line in result.summary():
        print(line)
    print("=" * 68)

    if args.trades and result.trade_list:
        print("\ntrades:")
        for t in result.trade_list[: args.trades]:
            side = "long" if t.side > 0 else "short"
            print(
                f"  {t.symbol:<10} {side:<5} {t.entry:>12,.4g} -> {t.exit:>12,.4g}  "
                f"{t.pnl:>+10,.2f}  {t.reason[:44]}"
            )
    return 0


# -------------------------------------------------------------------------- run


def cmd_run(args: argparse.Namespace) -> int:
    from quantdesk.app.desk import Desk
    from quantdesk.data import build_feed

    settings = get_settings()
    _setup_logging(settings.log_level, quiet=not args.no_ui)

    offline = not args.live_data
    feed = build_feed(settings, offline=offline)

    broker = None
    if args.trade:
        # Placing orders needs credentials; market data does not. Fail loudly here
        # rather than silently running a desk that cannot trade.
        from quantdesk.execution import AlpacaBrokerError, AlpacaPaperBroker

        try:
            broker = AlpacaPaperBroker(
                api_key=settings.key() or "",
                secret_key=settings.secret() or "",
                paper=True,
                asset_class=settings.asset_class,
            )
        except AlpacaBrokerError as exc:
            print(f"cannot place orders: {exc}", file=sys.stderr)
            return 2
    else:
        from quantdesk.execution import SimBroker

        broker = SimBroker(
            commission_bps=settings.commission_bps,
            slippage_bps=settings.slippage_bps,
        )

    desk = Desk(
        settings=settings,
        broker=broker,
        feed=feed,
        mode="ALPACA PAPER" if args.trade else "SIMULATED",
    )

    if args.no_ui:
        return asyncio.run(_run_headless(desk))
    return _run_with_ui(desk)


async def _run_headless(desk) -> int:
    print(f"warming up on {desk.settings.warmup_bars} bars...")
    await desk.warmup()
    print(f"desk running in {desk.mode} mode. ctrl-c to stop.")
    try:
        await desk.run()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        await desk.stop()
    print("\n".join(f"{k}: {v}" for k, v in desk.status().items()))
    return 0


def _run_with_ui(desk) -> int:
    from quantdesk.ui import QuantDeskApp

    app = QuantDeskApp(desk)

    async def boot() -> None:
        # Warm up and then run the feed alongside the UI. Both live in the same event
        # loop, so a blocking call in either would stall the other - which is why the
        # broker adapter pushes its synchronous SDK calls to threads.
        await desk.warmup()
        desk._task = asyncio.create_task(desk.run())

    async def runner() -> None:
        await asyncio.gather(app.run_async(), boot())

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    return 0


# --------------------------------------------------------------------- validate


def cmd_validate(args: argparse.Namespace) -> int:
    """Backtest, then ask the statistics whether the result means anything."""
    import numpy as np

    from quantdesk.backtest import Backtester
    from quantdesk.data import build_feed
    from quantdesk.psychology import DisciplineMonitor
    from quantdesk.quant import analyse, bootstrap_sharpe, deflated_sharpe, fit_vol_regime

    settings = get_settings()
    symbols = args.symbols or settings.universe
    feed = build_feed(settings, offline=args.offline)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    history = feed.history(start=start, end=end, symbols=symbols)
    if not any(history.values()):
        print("no bars returned. try --offline.", file=sys.stderr)
        return 1

    result = Backtester(settings).run(history)
    print("=" * 72)
    print("BACKTEST")
    for line in result.summary():
        print(f"  {line}")

    equity = np.array([p.equity for p in result.equity_curve], dtype=float)
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]
    # Bars where the desk was flat contribute exactly zero and are not observations of
    # the strategy's behaviour. Left in, they dominate the distribution and make every
    # shape statistic describe the flatness rather than the trading.
    active = returns[np.abs(returns) > 0.0]

    print()
    print("IS THE EDGE REAL?")
    if returns.size < 30:
        print("  too few observations to say anything")
    else:
        boot = bootstrap_sharpe(returns, settings.bars_per_year, resamples=args.resamples)
        print(f"  {boot.line('sharpe')}")
        shape = analyse(returns)
        dsr = deflated_sharpe(
            boot.statistic,
            trials=args.trials,
            sample_size=returns.size,
            skew=shape.skew,
            excess_kurtosis=shape.excess_kurtosis,
        )
        print(
            f"  deflated sharpe probability {dsr:.3f} assuming {args.trials} "
            f"parameter trial(s)"
        )
        if not boot.significant:
            print("  VERDICT: no demonstrated edge. The point estimate is not evidence.")
        elif dsr < 0.5:
            print("  VERDICT: significant, but likely a selection artefact once")
            print("           the number of variants tried is accounted for.")
        else:
            print("  VERDICT: edge survives resampling and selection adjustment.")

        print()
        print(
            f"RETURN DISTRIBUTION  ({active.size} active bars of {returns.size}; "
            f"flat bars excluded)"
        )
        if active.size >= 30:
            for line in analyse(active).summary():
                print(f"  {line}")

            print()
            print("VOLATILITY REGIME")
            for line in fit_vol_regime(active).summary():
                print(f"  {line}")
        else:
            print("  too few active bars to characterise")

    # Trade-level edge statistics, in R. Only trades whose planned risk is known can
    # contribute: without a denominator, R is not defined, and feeding those in as 0.0R
    # would count them as scratches and pull the average toward zero.
    monitor = DisciplineMonitor()
    for trade in result.trade_list:
        if trade.risk_known:
            monitor.on_exit(trade.symbol, r_multiple=trade.r_multiple, pnl=trade.pnl)
    print()
    print("RISK PER TRADE")
    print(
        f"  median {result.median_risk:,.2f}  range {result.min_risk:,.2f} to "
        f"{result.max_risk:,.2f}  spread {result.risk_dispersion:,.0f}x"
    )
    print(
        f"  expectancy {result.risk_weighted_r:+.3f}R risk-weighted, "
        f"{result.mean_r:+.3f}R unweighted"
    )
    if result.trades_without_risk:
        print(
            f"  {result.trades_without_risk} of {result.trades} trades had no recorded "
            f"risk and are excluded"
        )
    if not result.r_is_trustworthy:
        print("  The two expectancy figures differ because risk per trade is uneven.")
        print("  Only the risk-weighted figure can be compared with the P&L; the")
        print("  unweighted one is an average over incomparable denominators and is")
        print("  dominated by whichever trades happened to risk least.")

    print()
    print("EDGE AND DISCIPLINE")
    if not result.r_is_trustworthy:
        print("  (unweighted R - read alongside the warning above)")
    for line in monitor.summary():
        print(f"  {line}")
    print("=" * 72)
    return 0


# ------------------------------------------------------------------------- scan


def cmd_scan(args: argparse.Namespace) -> int:
    """Run the analysis once and print Murphy's checklist. No trading."""
    from quantdesk.analysis.checklist import TechnicalChecklist
    from quantdesk.analysis.symbol import SymbolAnalysis
    from quantdesk.data import build_feed
    from quantdesk.pipeline import _base_timeframe
    from quantdesk.analysis.timeframes import Timeframe

    settings = get_settings()
    symbols = args.symbols or settings.universe
    feed = build_feed(settings, offline=args.offline)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    history = feed.history(start=start, end=end, symbols=symbols)

    checklist = TechnicalChecklist()
    base = _base_timeframe(settings.timeframe)
    exit_code = 0
    for symbol, bars in history.items():
        if not bars:
            print(f"{symbol}: no data")
            exit_code = 1
            continue
        sa = SymbolAnalysis(
            symbol, timeframes=[base, Timeframe.H1, Timeframe.H4], base_timeframe=base
        )
        for bar in bars:
            sa.update(bar)
        verdict = checklist.evaluate(sa)
        print("=" * 74)
        for line in verdict.summary():
            print(line)
        print()
    return exit_code


# ---------------------------------------------------------------------- history


def cmd_history(args: argparse.Namespace) -> int:
    """Fetch long-run history and cache it locally."""
    from quantdesk.data import BarCache, HistoryError, HistoryStore
    from quantdesk.data.history import ShillerSource, YahooChartSource, resample_monthly

    settings = get_settings()
    store = HistoryStore(BarCache(settings.db_path))

    if args.source == "shiller":
        source = ShillerSource()
        symbol = args.symbol or "SP500"
    else:
        source = YahooChartSource(interval=args.interval)
        symbol = args.symbol or "^GSPC"

    print(f"fetching {symbol} from {source.name} ({source.timeframe})...")
    try:
        bars = store.load(source, symbol, refresh=args.refresh)
    except HistoryError as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1

    if not bars:
        print("no bars returned", file=sys.stderr)
        return 1

    years = (bars[-1].ts - bars[0].ts).days / 365.25
    print(
        f"{len(bars):,} bars  {bars[0].ts:%Y-%m-%d} to {bars[-1].ts:%Y-%m-%d}  "
        f"({years:.1f} years)"
    )
    print(f"first close {bars[0].close:,.2f}   last close {bars[-1].close:,.2f}")

    no_volume = sum(1 for b in bars if b.volume <= 0)
    if no_volume:
        print(
            f"note: {no_volume:,} bars carry no volume ({no_volume / len(bars):.0%}); "
            f"volume filters cannot work on those"
        )

    if args.monthly:
        monthly = resample_monthly(bars)
        print(
            f"resampled to {len(monthly):,} complete months "
            f"(the unfinished month is excluded)"
        )

    if args.source == "shiller":
        records = source.records()
        with_cape = [r for r in records if r.cape is not None]
        if with_cape:
            print(
                f"CAPE for {len(with_cape):,} months from {with_cape[0].ts:%Y-%m}, "
                f"latest {with_cape[-1].cape:.1f} at {with_cape[-1].ts:%Y-%m}"
            )
        print("note: Shiller prices are monthly averages, not tradeable OHLC")
    return 0


# ------------------------------------------------------------------------ perps


def cmd_perps(args: argparse.Namespace) -> int:
    """Funding rate and open interest for a perpetual future."""
    from quantdesk.data.perps import (
        PerpsError,
        default_source,
        open_interest_signal,
        summarise_funding,
    )

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    try:
        source = default_source(args.venue)
        print(f"fetching {args.symbol} from {source.name} ({args.days}d)...")
        rates = source.funding(args.symbol, start=start, end=end)
        interest = source.open_interest(args.symbol, period=args.period)
    except PerpsError as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1

    if not rates:
        print("no funding history returned", file=sys.stderr)
        return 1

    print()
    print("FUNDING")
    for line in summarise_funding(rates, args.symbol).summary():
        print(f"  {line}")

    print("  largest settlements:")
    for r in sorted(rates, key=lambda r: abs(r.rate), reverse=True)[:5]:
        payer = "longs paid" if r.longs_pay else "shorts paid"
        print(
            f"    {r.ts:%Y-%m-%d %H:%M}  {r.rate * 100:+.4f}%  "
            f"({r.annualised:+.1%} annualised, {payer})"
        )

    print()
    print("OPEN INTEREST")
    if len(interest) < 2:
        print("  not enough history returned to read a trend")
        print("  note: venues keep only a short window of open interest; a long series")
        print("        has to be accumulated forward, not backfilled")
        return 0

    print(
        f"  {len(interest)} points  {interest[0].ts:%Y-%m-%d} to "
        f"{interest[-1].ts:%Y-%m-%d}"
    )
    print(f"  {interest[0].contracts:,.0f} -> {interest[-1].contracts:,.0f} contracts")
    marks = [r.mark_price for r in rates if r.mark_price is not None]
    if marks:
        reading, why = open_interest_signal(marks, [o.contracts for o in interest])
        print(f"  reading: {reading}")
        print(f"           {why}")
    else:
        print("  this venue does not report a mark price with funding, so price and")
        print("  open interest cannot be read together here. Try --venue binance.")
    return 0


# --------------------------------------------------------------------- research


def cmd_research(args: argparse.Namespace) -> int:
    """Interactive terminal for inspecting backtests, long history, and perps.

    Read-only. Needs no keys, no live desk, and places no orders.
    """
    from quantdesk.ui.research import ResearchApp

    _setup_logging(get_settings().log_level, quiet=True)
    ResearchApp().run()
    return 0

# ---------------------------------------------------------------------- paper


def cmd_paper(args: argparse.Namespace) -> int:
    """Paper trading terminal: live perps prices, simulated money, charts.

    Prices, funding and open interest are real. Orders are matched against those prices
    by the paper broker and settle against a demo balance chosen in the terminal. No
    credential path exists here, so no real order can be placed.
    """
    from quantdesk.ui.trading import TradingTerminal

    _setup_logging(get_settings().log_level, quiet=True)
    TradingTerminal().run()
    return 0

# ------------------------------------------------------------------------- web


def cmd_web(args: argparse.Namespace) -> int:
    """Browser terminal for paper trading against the live market.

    Bound to loopback only. There is no authentication, so the bind address is the
    access control; it is deliberately not settable from the command line.
    """
    from quantdesk.web.server import serve

    _setup_logging(get_settings().log_level, quiet=True)
    return serve(port=args.port, open_browser=not args.no_browser)

# ----------------------------------------------------------------------- doctor


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what is configured and what is missing, without guessing."""
    settings = get_settings()
    print("configuration")
    print(f"  universe        {', '.join(settings.universe)}")
    print(f"  asset class     {settings.asset_class}")
    print(f"  timeframe       {settings.timeframe} ({settings.timeframe_minutes}m)")
    print(f"  starting equity {settings.starting_equity:,.0f}")
    print(f"  leverage        {settings.leverage}")
    print(f"  data dir        {settings.data_dir}")

    print("\nmurphy settings")
    print(f"  agents enabled      {settings.enable_murphy_agents}")
    print(f"  timeframe veto      {settings.require_timeframe_agreement}")
    print(f"  reward:risk gate    {settings.min_reward_risk}:1")
    print(f"  risk per trade      {settings.max_risk_per_trade_pct:.0%}")

    print("\ncredentials")
    has_key = bool(settings.key() and settings.secret())
    print(f"  alpaca keys     {'present' if has_key else 'MISSING'}")
    print("                  market data works without them; orders do not")

    print("\ndependencies")
    for module, purpose in (
        ("alpaca", "market data and paper orders"),
        ("textual", "terminal UI"),
        ("pandas", "data handling"),
        ("numpy", "indicator maths"),
    ):
        try:
            __import__(module)
            print(f"  {module:<12} ok        ({purpose})")
        except ImportError:
            print(f"  {module:<12} MISSING   ({purpose})")

    print("\nagents")
    from quantdesk.agents import build_roster

    signals, regimes = build_roster(settings)
    for agent in signals:
        print(f"  {agent.name:<22} weight {agent.weight:<5} warmup {agent.warmup}")
    for agent in regimes:
        print(f"  {agent.name:<22} (regime)")
    return 0


# -------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quantdesk",
        description="A small systematic trading desk built on Murphy's technical analysis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="replay history through the pipeline")
    bt.add_argument("--days", type=int, default=30)
    bt.add_argument("--symbols", nargs="*", default=None)
    bt.add_argument("--offline", action="store_true", help="use the synthetic feed")
    bt.add_argument("--trades", type=int, default=0, help="print this many trades")
    bt.set_defaults(func=cmd_backtest)

    run = sub.add_parser("run", help="live paper session with the terminal UI")
    run.add_argument(
        "--trade", action="store_true",
        help="place orders on Alpaca paper (needs keys). Without it, orders are simulated.",
    )
    run.add_argument(
        "--live-data", action="store_true",
        help="stream real Alpaca bars instead of the synthetic feed",
    )
    run.add_argument("--no-ui", action="store_true", help="headless, log to stderr")
    run.set_defaults(func=cmd_run)

    val = sub.add_parser(
        "validate", help="backtest, then test statistically whether the edge is real"
    )
    val.add_argument("--days", type=int, default=365)
    val.add_argument("--symbols", nargs="*", default=None)
    val.add_argument("--offline", action="store_true")
    val.add_argument("--resamples", type=int, default=2000)
    val.add_argument(
        "--trials", type=int, default=1,
        help="how many parameter variants were tried, for the deflated Sharpe",
    )
    val.set_defaults(func=cmd_validate)

    scan = sub.add_parser("scan", help="print Murphy's checklist for each symbol")
    scan.add_argument("--days", type=int, default=14)
    scan.add_argument("--symbols", nargs="*", default=None)
    scan.add_argument("--offline", action="store_true")
    scan.set_defaults(func=cmd_scan)

    hist = sub.add_parser("history", help="fetch and cache long-run history")
    hist.add_argument(
        "--source", choices=["yahoo", "shiller"], default="yahoo",
        help="yahoo: daily index bars from 1928. shiller: monthly S&P from 1871.",
    )
    hist.add_argument(
        "--symbol", default=None,
        help="defaults to ^GSPC for yahoo, SP500 for shiller",
    )
    hist.add_argument("--interval", choices=["1d", "1wk", "1mo"], default="1d")
    hist.add_argument("--monthly", action="store_true", help="also resample to months")
    hist.add_argument("--refresh", action="store_true", help="ignore the local cache")
    hist.set_defaults(func=cmd_history)

    perps = sub.add_parser("perps", help="funding rate and open interest for a perp")
    perps.add_argument("--symbol", default="BTC/USD")
    perps.add_argument("--venue", choices=["binance", "bybit"], default="binance")
    perps.add_argument("--days", type=int, default=90)
    perps.add_argument("--period", default="1d", help="open interest bucket, e.g. 1d, 4h")
    perps.set_defaults(func=cmd_perps)

    res = sub.add_parser(
        "research", help="interactive terminal: risk accounting, long history, perps"
    )
    res.set_defaults(func=cmd_research)
    paper = sub.add_parser(
        "paper", help="paper trading terminal: live perps prices, demo money, charts"
    )
    paper.set_defaults(func=cmd_paper)
    web = sub.add_parser(
        "web", help="browser terminal: live perps charts, demo money, on localhost"
    )
    web.add_argument("--port", type=int, default=8787)
    web.add_argument("--no-browser", action="store_true", help="do not auto-open")
    web.set_defaults(func=cmd_web)
    doc = sub.add_parser("doctor", help="report configuration and missing pieces")
    doc.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    if args.command != "run":
        _setup_logging(get_settings().log_level)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
