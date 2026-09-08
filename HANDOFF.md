# QuantDesk — Session Handoff

**State:** 12 of 17 planned tasks complete. 135 tests passing. The system runs
end-to-end on real market data. Tasks 15, 16, 17 remain, plus one open investigation
that must be resolved before any performance claim is made.

---

## 1. What this is

A small systematic trading desk ("Jane Street but small") at `v:\iiui\quantdesk`.
Requirements as originally stated by the user, recovered from the conversation export
at `v:\iiui\quantdesk-conversation-clean.json`:

1. Algo trading setup with multiple AI agents that continuously scan the market, find
   opportunities and trade on demo money, with a **Bloomberg-like terminal**.
2. Implement Murphy's *Technical Analysis of the Financial Markets* as the analytical
   brain — "apply all the technicals and tricks and analysis from the book".
3. Mathematical / probability / statistical models — "quant analysis rather than human
   market intuition". Fetch **100 years of market data** and **perps data**, backtest,
   and build models on it.
4. After Murphy, a **deep ML / AI agent** that executes on those analyses and models.
5. Order the user set: Murphy → *Trading in the Zone* → ML agent → quant models +
   long history + perps.

Items 1, 2 and the *Trading in the Zone* layer are done. Item 3 is done except the
data acquisition. Item 4 is not started.

---

## 2. Environment

- Python 3.10.6, venv at `v:\iiui\quantdesk\.venv`
- Installed: `alpaca-py 0.44.0`, `textual 8.2.8`, `pandas 2.3.3`, `numpy 2.2.6`,
  `pydantic 2.12.5`, `pydantic-settings 2.13.1`, `rich 14.3.3`, `pypdf 6.6.0`,
  `pytest 8.3.4`, `scipy 1.15.1`, `scikit-learn 1.6.1`
- **Alpaca crypto bars work with no API key.** `backtest`, `scan` and `validate` all
  run on a bare checkout. Keys are only needed to place orders.
- Books extracted: Murphy at `_book_epub.txt` + `_ch/*.txt` (22 chapters);
  *Trading in the Zone* at `_zone.txt` (74k words, extracted this session).

### Tooling gotchas that cost time — read these

- **`fs_write` over ~12KB silently misreports.** A 20KB write lands on disk correctly
  but the tool returns "aborted by the user". Keep writes under ~12KB and **verify on
  disk afterwards**; do not trust the tool's success/failure report.
- **PowerShell `.Replace()` on file contents fails** because of `\n` vs `\r\n`. Use the
  `str_replace` tool, or PowerShell's `-replace` operator with `Set-Content -NoNewline`.
- `permissions.yaml` has no `fs_write` rule, so writes prompt. Adding
  `- capability: fs_write` / `effect: allow` removes the prompt. User has not confirmed.

---

## 3. Commands

```powershell
cd v:\iiui\quantdesk
.venv\Scripts\python.exe -m pytest tests/ -q                    # 193 tests

.venv\Scripts\python.exe -m quantdesk.cli doctor                # config + deps + agents
.venv\Scripts\python.exe -m quantdesk.cli scan --days 14        # Murphy's checklist
.venv\Scripts\python.exe -m quantdesk.cli backtest --days 30
.venv\Scripts\python.exe -m quantdesk.cli validate --days 365 --trials 12
.venv\Scripts\python.exe -m quantdesk.cli run                   # terminal UI, simulated
.venv\Scripts\python.exe -m quantdesk.cli run --live-data --trade  # needs Alpaca keys
.venv\Scripts\python.exe -m quantdesk.cli paper                  # PAPER TRADING: live prices, demo money
.venv\Scripts\python.exe -m quantdesk.cli research               # research terminal, no keys
.venv\Scripts\python.exe -m quantdesk.cli history --source yahoo --monthly
.venv\Scripts\python.exe -m quantdesk.cli history --source shiller
.venv\Scripts\python.exe -m quantdesk.cli perps --symbol BTC/USD --venue binance --days 90
```
`paper` is the headline command: set a demo balance, pick perps, and trade live Binance
prices with simulated money. Charts, funding, positions and a running result. No credential
path exists in it, so no real order can be placed.

`research` is the other one worth knowing: three tabs over risk accounting, long history
and perps funding. Read-only, no keys, no orders, no live desk.

Config is env-driven with a `QD_` prefix, e.g. `$env:QD_TIMEFRAME="1Hour"`.
Murphy's methods suit hourly and daily bars far better than 5-minute; see §6.

---

## 4. Architecture

The important structural property: **`Pipeline` is the single decision path, and both
the backtester and the live desk run that same object.** They differ only in which
clock and which broker are injected. There is no separate simulation strategy code to
drift out of sync.

Per-bar sequence, in dependency order:

```
bar
 -> SymbolAnalysis.update()        indicators, pivots, trend, levels, trendlines,
                                   gaps, retracement, patterns, candles, Elliott,
                                   timeframes, divergence
 -> TechnicalChecklist.evaluate()  Murphy ch19, 23 questions -> Verdict
 -> agents (10)                    each sees the SAME analysis + verdict
 -> TradeManager.decide()          STOPS FIRST, before any new signal is considered
 -> SignalBlender.blend()          -> target weight
 -> MoneyManager.plan()            Murphy ch16 limits -> size, stop, objective
 -> RiskManager.check_order()      account-level veto
 -> orders
```

### Module map

| Area | Files | Notes |
|---|---|---|
| Murphy structure | `analysis/{swings,dow,levels,trendlines,retracement,gaps}.py` | built in earlier sessions |
| Chart patterns | `analysis/patterns/` (10 files) | 20 formations, ch5-6 |
| Candlesticks | `analysis/candles/` (6 files) | ~35 patterns, ch12 |
| Elliott | `analysis/elliott/` (3 files) | ch13, 3 hard rules enforced |
| Multi-timeframe | `analysis/timeframes.py` | aggregation + top-down gate |
| Integration | `analysis/symbol.py` | owns the whole stack |
| Synthesis | `analysis/checklist.py` | ch19, all 23 questions |
| Money mgmt | `risk/money.py` | ch16 limits verbatim |
| Agents | `agents/{murphy,formations}.py` | 6 Murphy agents + 4 statistical |
| Decision | `pipeline.py`, `trade_manager.py` | shared by backtest and live |
| Execution | `execution/{alpaca_broker,router}.py` | paper-only, one order per symbol |
| Backtest | `backtest/{engine,metrics,account}.py` | next-bar-open fills |
| Terminal | `ui/{app,panels}.py`, `ui/theme.tcss` | Textual, amber on black |
| Runtime | `app/desk.py`, `cli.py` | warmup, pause, kill |
| Psychology | `psychology/` (3 files) | Douglas, as invariants |
| Quant | `quant/` (4 files) | distributions, validation, regime, Kelly |

### Design decisions worth preserving

- **Stops are inviolable.** `TradeManager` checks stops before signals. A stop a fresh
  signal can override is not a stop, and the risk cap sized against it becomes fiction.
  `stop_hit` uses the bar's low/high, not the close. A gap through the stop fills at the
  **open**, not the stop price.
- **Discrete trade management, not continuous rebalancing.** Rebalancing a Murphy-style
  position every bar produced 9 fills per trade and fees worth 55% of the loss.
- **Trend context is mandatory for candles.** `CandleReader.update(bar, prior_trend)`
  refuses a bullish reversal in an uptrend, per Murphy: "You cannot have a bullish
  reversal pattern in an uptrend."
- **Elliott returns ranked candidates with violations attached**, never one confident
  answer. Wave counting is the most subjective technique in the book.
- **The checklist marks 4 of 23 items UNAVAILABLE** honestly: no sentiment feed
  (item 16), cycles ch14 not built (20, 21), point and figure ch11 not built (23's
  P&F half). `Verdict.coverage` reports how much was actually answered.
- **Checklist weights are declared as our judgement, not Murphy's.** He assigns none.
- **The UI owns no trading state.** It reads the desk on a timer and every panel render
  is wrapped, so a formatting bug cannot stop orders being managed.
- **`AlpacaPaperBroker` refuses `paper=False` in its constructor.** Live trading is
  deliberately not implemented.
- **Detector errors are recorded, never swallowed.** A silently skipped detector looks
  identical to a quiet market. `PatternScanner(strict=True)` raises; otherwise
  `.errors` accumulates. A real `TypeError` hid here for an entire session.

---

## 5. Verified results on real data

30 days and 365 days of real Alpaca crypto bars, BTC/USD and ETH/USD, no API key.

| Run | Return | Sharpe | Hit rate | Profit factor | Fills |
|---|---|---|---|---|---|
| 5Min, 30d, first attempt | -1.79% | -5.08 | 38.1% | 0.71 | 4020 |
| 5Min, 30d, after trade manager | -1.84% | -4.44 | 33.2% | 0.76 | 2271 |
| 1Hour, 365d | -1.56% | -0.37 | 45.3% | 0.94 | 2240 |
| 1Hour, 365d, leverage 3 | -6.51% | -0.76 | 45.8% | 0.88 | 2241 |
| 1Hour, 365d, leverage 5 | -12.89% | -1.46 | 44.2% | 0.77 | 1840 |

`validate` on the 1Hour/365d run reports:

```
sharpe -0.366  95% CI [-1.672, +1.035]  p=0.7020  NOT distinguishable from zero
deflated sharpe probability 0.000 assuming 12 parameter trial(s)
VERDICT: no demonstrated edge. The point estimate is not evidence.
```

Return distribution over 9,943 active bars: skew +1.27, excess kurtosis +67,
Jarque-Bera p=0 (not normal), real 99% VaR 1.22x worse than a normal model predicts.
Volatility regime fit converges cleanly at a 6.4x ratio between calm and turbulent.

**The honest position: the system is correct and working; it does not have a
demonstrated edge.** Those are different claims and only the first has been established.
Do not tune parameters to make the backtest look better — with 12+ variants already
tried the deflated Sharpe is already 0.000, and further tuning is overfitting, not
research.

---

## 6. RESOLVED — the R-multiple artefact

`validate` reported **+1.02R expectancy with a 3.39:1 payoff and a +22.5% edge margin**
while the money was **negative**. Both could not be true. Root cause found, confirmed by
measurement, and fixed.

### The earlier diagnosis in this document was wrong

It named `Backtester._track_trade` as the prime suspect, on the theory that risk was being
reconstructed incorrectly from fills. That reconstruction *was* sloppy and has been fixed,
but it was not the cause. Measurement showed `plan.risk_amount` — the money manager's own
intended risk, computed before the backtester touches anything — already had median **43**
against a 500 target. The backtester was faithfully recording a number that sizing had
already got wrong.

What had been established, and still holds:

- Recorded `initial_risk` per trade has median 54 against a 500 target, ranging
  **0.10 to 269** — a 2,700x spread.
- A few trades with near-zero recorded risk inflate the *mean* R enormously, because
  `r_multiple = pnl / initial_risk` explodes as the denominator approaches zero.
- Adding `risk_target_pct` (0.5% of equity) did **not** change the R statistics at all,
  which falsified the first hypothesis that risk simply was not being targeted.
- Raising leverage made returns monotonically **worse** (-1.77% → -6.51% → -12.89%)
  while the money payoff ratio stayed near 1.05:1 at every level. So the 3.39:1 R
  payoff is an artefact, not a real asymmetry being squandered by sizing.

### Actual root cause

Tallying `plan.binding_limit` over a run settled it:

| count | binding limit | risk min | median | max |
|---|---|---|---|---|
| 659 | 10% market commitment | 0.11 | 36.09 | 315.53 |
| 144 | 0.50% risk target | 50.43 | 103.60 | 362.42 |
| 3 | 20% group margin | 25.09 | 51.91 | 52.10 |

**82% of trades were sized by the 10% market-commitment ceiling, not the risk target.**
At leverage 1.0, `margin_per_unit = entry / leverage = entry`, so the commitment ceiling
allows `10% × equity / entry` units. Actual risk per trade is then that quantity times the
stop distance — which varies with ATR across roughly three orders of magnitude. Risk per
trade was never equalised, so `r_multiple = pnl / initial_risk` had a denominator spanning
~3,000x, and the plain **mean** of R was dominated by whichever trades happened to risk
least.

A second, compounding effect: `conviction` and pyramid decay multiply `qty` *after*
`min(limits)` picks the binding ceiling, so even the risk-target branch delivered well
under target (50 to 362, not 500) while still reporting "risk target" as the binding limit.

This explains every earlier observation, including both hypotheses recorded as falsified:

- Adding `risk_target_pct` changed nothing **because the risk target almost never binds**.
- Raising leverage made returns worse because leverage relaxes `margin_per_unit`, letting
  the risk target bind, which increases size and amplifies a losing strategy. The money
  payoff stayed ~1.05:1 because trade quality never changed.

### What was changed

The commitment ceiling is correct and must not be violated — Murphy's 10-15% is a hard
limit. So the fix does not force risk to be equal; it makes the inequality visible and the
reported statistics honest about it.

- **`Order.planned_risk`** carries currency at risk from decision time to fill.
  `Backtester` stores it per unit of quantity and each fill takes its share, so partial
  fills, pyramid layers and flip residuals all account correctly. Exits carry zero.
  Verified by a conservation test: `sum(order.planned_risk) == sum(trade.initial_risk)`.
- **`PositionPlan.risk_target`** and **`risk_delivered`** record what the trade intended to
  risk versus what the ceilings allowed. That gap was previously invisible.
- **`binding_limit`** now names conviction or pyramid decay when either is what actually
  cut the size, rather than naming a ceiling that was not the operative constraint. Kept
  low-cardinality so it stays groupable.
- **`BacktestResult.risk_weighted_r`** — total P&L over total risk. The only R aggregate
  that cannot disagree in sign with the money, because it *is* the money over the risk.
  Reported next to `mean_r`, with median/min/max risk and a `risk_dispersion` ratio.
- **`r_is_trustworthy`** is false above 10x dispersion, and both `summary()` and `validate`
  print an explicit warning saying which figure to use.
- **`TradeRecord.peak_qty`** — `qty` is decremented to zero as a position closes, so every
  closed trade previously reported size 0 and attribution was impossible.
- **`TradeRecord.risk_known`** — trades with no recorded risk are excluded from R rather
  than counted as 0.0R scratches, which had been pulling expectancy toward zero.

### Current reading

A 20-day two-symbol offline run: return +4.45%, risk-weighted **+0.085R**, unweighted
**-0.171R**, dispersion 8,545x, zero trades missing risk. The two figures differ by design
and the warning fires. The risk-weighted figure agrees in sign with the P&L, which is now
covered by a test.

**The offline synthetic feed is a random walk and is not evidence of an edge.** The
real-data runs in §5 were negative and nothing here changes that. What changed is that the
reported statistics are now internally consistent.

### Still open, and deliberately not "fixed"

Risk per trade remains uneven, because equalising it would require either violating the
10% commitment ceiling or trading with leverage. Both are policy decisions, not bugs. The
options, for whoever picks this up:

1. **Accept it** and read only `risk_weighted_r`. Cheapest, and correct as it stands.
2. **Raise leverage** so the risk target binds. Makes R meaningful but increases size, and
   §5 already showed that amplifies the losses.
3. **Refuse trades that cannot reach the risk target**, e.g. skip when
   `risk_delivered < 0.5`. Equalises risk, at the cost of far fewer trades.

Option 1 is in force. Nothing selects between the other two without an edge to protect.

---

## 7. Bugs found and fixed this session

Kept because each was found by a test or a real-data run, and each would silently
return.

1. **`ATR.value` and `RollingWindow.mean` are properties, not methods.** Calling them as
   methods raised `TypeError` mid-completion, which the scanner's blanket
   `except Exception` swallowed — leaving patterns permanently stuck with
   `objective=0`. Fixed, and the blanket except was replaced with a recorded error list
   plus a `strict` flag.
2. **`MoneyManager.plan` returned incremental room** while the pipeline treated it as the
   target size, so once a position reached the 10% cap every subsequent bar refused with
   "limits leave no tradable size". Now sizes the target, excluding the symbol's own
   existing commitment from the portfolio and group ceilings.
3. **`RiskManager.check_order` wants exposure as fractions of equity**, not currency.
   Passing notional compared thousands of dollars against a cap of 1.0 and vetoed
   everything.
4. **`day_start_equity` never rolled over**, turning the daily loss limit into a
   *lifetime* limit — one 3% drawdown put the desk permanently into reduce-only, with
   1,740 identical vetoes as the symptom. `Pipeline._roll_day()` added.
5. **Breakeven ratchet at 1.0R converted winners into scratches**, producing a 1:1
   payoff against a 3:1 entry gate. Moved to 1.5R with a 1.5R trail measured on the high
   water mark.
6. **`_stop_and_target` selected patterns whose objective had already been passed**,
   putting the target behind the entry — 731 "stop on wrong side" refusals.
7. **Murphy's 10-15% is margin, not notional** ("available for margin deposit"), so
   `plan()` takes `leverage`. At leverage 1 the commitment cap binds and risk lands near
   1%; at leverage 10 the 5% risk cap binds as he intends.
8. **`RetracementMap.retracement_of` used `abs()`**, conflating a retracement with an
   extension — two opposite readings collapsed into one number. Added
   `signed_retracement_of()`.
9. **Massive pattern over-detection** on a random walk: 1,632 patterns per 6,000 bars,
   including 395 broadening formations that Murphy calls "relatively rare". Fixed by
   gating re-scans on pivot changes, enforcing strict monotonicity of reversal points
   instead of average slope, adding a containment test, scoring the best window instead
   of first-fit, and overlap suppression. Result: 537 total, symmetrical triangles
   326 → 40, broadening → 0.
10. **`Pattern.key` omitted `start_index`**, so every flag collapsed into a single key.
11. **Wilson interval replaced the normal approximation** in `EdgeStats`, which claimed
    certainty from 9 wins out of 9.
12. **`fit_vol_regime` produced a 296-million-x volatility ratio** on strategy returns,
    because a flat desk emits exact zeros and the mixture collapsed onto them. Added
    `drop_zeros` and a `degenerate` flag with a capped ratio.
13. **Hurst was being computed on returns**, a category error that reads near zero for
    any return series. Now computed on the cumulative path. Also documented that Hurst
    is drift-invariant: a random walk plus drift is still 0.5.
14. **`kelly_position` checked the edge before checking reliability**, so a 5-trade
    sample was judged rather than deferred to the fixed cap.
15. **Risk per trade was aggregated with a plain mean over denominators spanning
    ~3,000x**, so reported R expectancy could be positive while the account lost money.
    See §6. Fixed by carrying risk on the order and adding a risk-weighted aggregate.
16. **`TradeRecord.qty` was always 0 on closed trades** — it is the live remaining
    quantity and gets decremented to zero before the record is filed, so size attribution
    silently returned nothing. `peak_qty` added.
17. **`initial_risk` was never incremented on pyramid adds** and was left at 0.0 on flip
    residuals, so a pyramided trade measured R against its first layer alone and a flipped
    position measured R against nothing.
18. **`binding_limit` reported the ceiling even when conviction decided the size.**
    `conviction` and pyramid decay are applied after `min(limits)`, so a plan could report
    "0.50% risk target" while delivering 28% of it. This is why the risk target looked like
    it was binding when it was not.
19. **`BarCache` could not read any bar before 1970.** It used
    `datetime.fromtimestamp`, which raises `OSError: [Errno 22]` for negative epoch values
    on Windows. Every bar before 1970 has one, so a century of index history was exactly
    the case that broke. `core.types.from_epoch()` adds a timedelta to the epoch instead;
    10,496 pre-1970 `^GSPC` bars now round-trip.
21. **A simulated live session never filled anything.** `Desk._handle_bar` never called
    `broker.on_bar`, so `SimBroker` was never advanced and pending orders sat unmatched
    forever. Combined with `SimBroker.account()` returning `None`, desk equity stayed
    pinned at its opening value. The session looked like a strategy declining to trade,
    which is indistinguishable from one working correctly. `_tick_broker()` added, called
    before the decision so fills land at the bar open.
22. **`risk.can_open` is a property, called as `can_open()`.** Raised
    `TypeError: 'bool' object is not callable` on the first bar that produced an order,
    which killed the desk loop. The loop catches its own failures and carries on, so the
    symptom was a session that silently stopped trading rather than a traceback. Same
    class of bug as #1.
20. **`compute_metrics` raised `OverflowError` on short equity curves.** Annualising a
    two-bar sample of 5-minute crypto data raises the growth factor to the power of
    ~100,000, and `**` raises rather than returning infinity. Metrics crashed entirely
    instead of reporting a meaningless number as meaningless. Now computed in log space
    via `_annualise()`, returning infinity only when the extrapolation genuinely exceeds
    a float.

Two corrections to earlier session notes:

- Murphy does **not** prescribe increasing size after equity dips. He poses the streak
  question and says the answers "aren't as simple or obvious as they seem", giving only
  one firm warning: doubling up after a winning streak means "you'll wind up giving it
  all back". Encoded as `streak_growth_cap` only.
- A sine wave is **not** a mean-reverting test series for Hurst; it is smooth and reads
  as trending. Use an Ornstein-Uhlenbeck process.

---

## 8. Next steps

### 0. R-multiple artefact — done

Resolved; see §6. Performance can now be reported, using `risk_weighted_r`. The unweighted
figure is still printed but is labelled and should not be quoted on its own.

### Task 15 - long history and perps: DONE

`quantdesk/data/history.py` and `quantdesk/data/perps.py`. Every endpoint below was
probed live before anything was built on it, because the URLs recorded in an earlier
draft of this document had gone stale.

| source | coverage | resolution | key | status |
|---|---|---|---|---|
| Shiller `ie_data.xls` | Jan 1871 - Sep 2023, 1,833 months | monthly | no | works |
| Yahoo `^GSPC` chart | Dec 1927 - now, 24,787 bars | daily | no | works |
| Binance USD-M perps | ~Sep 2019 - now | funding + OI | no | works |
| Bybit linear perps | ~2019 - now | funding + OI | no | works |

Dead ends, so nobody retries them: `shillerdata.com/data/ie_data.xls` is a 404 (use the
Yale path); Stooq returns an HTML block page rather than CSV; FRED's `fredgraph.csv`
timed out; OKX is unreachable from this machine. Yahoo's `v7` download and quote
endpoints do need a cookie and crumb now, but the `v8` chart endpoint used here does not.

Things that will bite whoever touches this next:

- **Shiller's date column encodes October as `.1` and January as `.01`.** As floats those
  are 0.1 and 0.01, so reading the digits after the decimal point as text turns October
  into January. `_shiller_date()` multiplies the fraction by 100 and rounds. Verified by
  checking that per-month counts are balanced at 152-153 each.
- **Shiller prices are monthly averages of daily closes, not OHLC.** `fetch()` sets all
  four prices equal on purpose. Do not invent an intramonth range to fill it in; any
  stop-distance calculation on top of fabricated highs and lows is worthless.
- **The file needs `xlrd`** (pinned at 2.0.2) because it is pre-2007 `.xls`, which
  `openpyxl` cannot open. Imported lazily, so every other command runs without it.
- **The published Shiller file trails the present by a year or more.** It is a research
  dataset, not a feed. Call `coverage()` rather than assuming it reaches today.
- **22% of `^GSPC` bars carry zero volume** (pre-1962). Volume filters silently do
  nothing on those rather than failing loudly.
- **Binance ignores `startTime=0` on `fundingRate`** and answers with the newest page
  regardless, so reaching the start of history means walking forward in explicit windows.
  Both venue classes page, with a no-forward-progress guard against spinning.
- **Open interest retention differs sharply:** ~31 days on Binance, ~200 points on Bybit.
  A long series has to be accumulated forward, not backfilled.

`resample_monthly()` drops the incomplete trailing month. A month still in progress is
not a bar that existed at the time, and including it lets a backtest act on a close that
had not happened.

**Scope, stated plainly:** a century is real for equity indices and does not exist for
crypto. Spot is ~15 years old and perps ~9. Long-history models have to be developed on
index data and transferred, and that transfer is an assumption to test, not a detail.

### Paper trading terminal: DONE

`quantdesk paper`, in `quantdesk/ui/trading.py`. Set a demo balance, pick perps, press
START. Prices, funding and open interest are real and live from Binance USD-M futures;
orders are matched by `PaperBroker` against those prices and settle against the demo
balance. There is no credential path in the file and no venue write call, so no
configuration makes it place a real order.

New pieces:

- `execution/paper_broker.py` - `PaperBroker` composes `SimBroker` (matching) with the
  backtester's `Account` (ledger), and implements the full `Broker` interface including
  `account()` and `positions()`, which `SimBroker` does not. Composition rather than
  subclassing so fill logic exists once and a paper result is reproducible by a backtest.
  Also records an equity curve and closed round trips, which is what "results" means.
- `data/binance_feed.py` - `BinancePerpsFeed`, live perp bars, no key. Verified back to
  Sep 2019 for BTCUSDT. **Only closed bars reach the strategy**: the last kline Binance
  returns is still forming and its high, low and close all move, so it is filtered by
  close time and exposed separately via `forming()` for the chart only.
- `ui/charts.py` - candlesticks and an equity area chart drawn with Unicode blocks. No
  plotting dependency. A doji is given at least one row, or naive scaling rounds the body
  to zero and the bar silently vanishes. The equity chart fills between the line and the
  starting balance, not down to the canvas floor - filling to the floor paints a winning
  session entirely in the losing colour.

Verified against real data: a 6-day replay of live BTCUSDT and ETHUSDT 5-minute perp bars
on 25,000 demo produced 207 fills, 108 round trips, and equity of 24,861 - a 0.55% loss,
with fees at 31 against a pre-cost P&L of -108. Consistent with §5: the machinery works,
the strategy has no demonstrated edge.

Two engine bugs surfaced doing this, both recorded as #21 and #22. Both made a live
session look inert rather than broken, which is why they survived this long.

### Research terminal UI: DONE

`quantdesk research`, in `quantdesk/ui/research.py`. Three tabs. Read-only: no `Desk`, no
keys, no orders. The live terminal in `ui/app.py` polls a running desk, which is the wrong
shape for one-shot research pulls, so this is a separate app that reuses the palette.

- **RISK** runs a backtest and shows both expectancy figures side by side, the risk
  dispersion, and a table of the 25 largest-|R| trades with each denominator visible. It
  prints `DEFECT` if risk-weighted R ever disagrees in sign with the P&L, which is
  arithmetically impossible and therefore a corruption check rather than a judgement.
- **HISTORY** loads Yahoo or Shiller cache-first, buckets by decade, and states the
  caveats on screen: the Shiller averaging, the zero-volume share, the dropped partial
  month, and the pre-1970 epoch handling.
- **PERPS** shows funding carry, the crowded-long read, largest settlements, and the
  price-versus-open-interest reading, with the retention limit stated.

Network and backtest work runs in `@work(thread=True, exclusive=True)` workers, so the UI
stays responsive and a second concurrent run of the same operation is refused. Every
worker reports failures into its own panel; a dead endpoint never takes the app down.

Requirements are at `.kiro/specs/research-terminal-ui/requirements.md`.

### Task 16 — ML / AI agent

Build `quantdesk/ml/`.

The feature pipeline is the valuable part and it already exists implicitly: every field
on `SymbolAnalysis` is a feature. Extract them into a flat vector.

- `ml/features.py` — `SymbolAnalysis` → feature vector. Must be computed from the same
  bar-by-bar path as live, or it leaks.
- `ml/dataset.py` — labels. Prefer triple-barrier labelling (target, stop, time limit)
  over fixed-horizon returns, because it matches how the desk actually exits.
- `ml/model.py` — gradient boosting from `scikit-learn` (already installed). Output
  **calibrated probabilities**, not scores, so they can be compared to the reward:risk
  gate directly.
- `ml/agent.py` — a `SignalAgent` wrapper. Should abstain when the predicted probability
  is near 0.5 rather than emitting a weak signal.

Non-negotiables, given what this session found:
- Train only with `quant.walk_forward_splits(..., gap=N)`. The gap is essential: features
  look back and labels look forward, so without it train and test share information.
- Every reported result goes through `quant.validation`. A model with a good in-sample
  score and an insignificant bootstrap Sharpe has learned noise.
- Feed the trial count into `deflated_sharpe`. Hyperparameter search inflates results and
  the correction is the only honest way to report it.

### Task 17 — Final integration and documentation

- `README.md`: what it is, install, the four commands, architecture diagram, and an
  explicit statement of what has and has not been demonstrated.
- Wire `DisciplineMonitor` into the live `Desk` so pauses and kills are recorded as
  interventions (the plumbing exists; it is not connected).
- Wire `quant.fit_vol_regime` risk scalar into the regime agent so measured volatility
  regime actually scales exposure.
- Add a `psychology` panel to the terminal showing the edge statistics and the streak
  verdict — the streak verdict is the most useful thing to have on screen during a
  drawdown.
- Run the full suite plus a live paper smoke test with real keys.

### Research direction, separate from building

The system currently has no edge. Things worth trying, in the order most likely to
matter, each validated with `quant.validation` and each counted as a trial:

1. **Slower timeframes.** Murphy's methods are built for daily and weekly charts and he
   says "master interday trading before trying intraday trading". 1Day has not been
   tested at all.
2. **Fees.** 1 bp commission plus 2 bps slippage was 55-77% of the loss in early runs.
   Fewer, larger trades change the arithmetic more than any signal improvement.
3. **Agent weights.** Ten agents currently vote. Test each in isolation to find which
   carry information; several probably carry none.
4. **Equity indices instead of crypto.** Murphy's framework was developed on markets with
   different microstructure, and long history is available there.

---

## 9. Task ledger

| # | Task | State |
|---|---|---|
| 1 | Extract Murphy ch08/12/13/16/19 parameters | done |
| 2 | Candlesticks (ch12) | done |
| 3 | Elliott wave (ch13) | done |
| 4 | Multi-timeframe (ch08) | done |
| 5 | Weight-of-evidence checklist (ch19) | done |
| 6 | Money management (ch16) | done |
| 7 | Murphy agents | done |
| 8 | Alpaca paper broker + router | done |
| 9 | Event-driven backtester | done |
| 10 | Textual terminal UI | done |
| 11 | Wire runnable app | done |
| 12 | End-to-end verification on real data | done |
| 13 | Trading in the Zone discipline layer | done |
| 14 | Quant / statistical models | done |
| — | R-multiple artefact | done (§6) |
| 15 | 100-year and perps data | done |
| -- | Research terminal UI | done |
| -- | Paper trading terminal (live prices, demo money) | done |
| 16 | ML / AI agent | not started |
| 17 | Final integration and docs | not started |

193 tests across 12 files: `test_patterns`, `test_candles_elliott`, `test_timeframes`,
`test_money`, `test_trade_manager`, `test_execution_backtest`, `test_psychology`,
`test_quant`, `test_research_ui`, `test_paper_trading`, `test_trading_ui`, plus
`conftest` fixtures (`make_bar`, `walk`, `flat`, `trending`).
The 13 added for §6 cover risk conservation from order to trade record, exits carrying no
risk, `peak_qty`, the risk-weighted/money sign agreement, and a synthetic case where one
near-zero denominator fools the unweighted mean but not the weighted one.
A further 14 in `test_research_ui` drive the research terminal through Textual's
`run_test()` harness. That is the only way this UI is verifiable here, since the dev
environment has no TTY. All are offline: the worker methods that reach the network are
never invoked, and the render methods are called directly with synthetic results.
Note for test authors: read a `Static`'s text via `widget.content`, not `.renderable`.
Textual 8 renamed it, and the wrong attribute fails identically for every panel, which
reads like a rendering bug rather than a test bug.

Note for test authors: `make_bar` defaults to `symbol="TEST"`. Trade manager tests must
use that symbol or the manager looks up nothing and every assertion fails confusingly.
