# QuantDesk

A small systematic trading desk: agent-based market analysis, risk-managed paper
execution, and terminals to inspect all of it. Built on John Murphy's *Technical
Analysis of the Financial Markets* for the analysis, and Mark Douglas's *Trading in
the Zone* for the discipline layer.

**Status: the machinery works and the strategy has no demonstrated edge.** Those are
separate claims and only the first is established. Backtests over real data are
negative, and the bootstrap Sharpe is not distinguishable from zero. Read
[HANDOFF.md](HANDOFF.md) before drawing conclusions from any number this produces.

No real orders. Execution is simulated against live prices; the Alpaca adapter is
paper-only and refuses live endpoints.

## Install

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Market data for crypto and perps needs no API key. Only Alpaca *order placement*
does, and that path is off by default.

## Commands

```powershell
.venv\Scripts\python.exe -m quantdesk.cli web        # browser terminal, charts, demo money
.venv\Scripts\python.exe -m quantdesk.cli paper      # same thing in the terminal
.venv\Scripts\python.exe -m quantdesk.cli research   # risk accounting, long history, perps
.venv\Scripts\python.exe -m quantdesk.cli backtest --days 30
.venv\Scripts\python.exe -m quantdesk.cli validate --days 365 --trials 12
.venv\Scripts\python.exe -m quantdesk.cli scan --days 14
.venv\Scripts\python.exe -m quantdesk.cli history --source yahoo --monthly
.venv\Scripts\python.exe -m quantdesk.cli perps --symbol BTC/USD --days 90
.venv\Scripts\python.exe -m quantdesk.cli doctor
```

`web` is the one to start with: set a demo balance, pick perps, and watch it trade
live Binance prices with simulated money. It binds to `127.0.0.1` only and has no
authentication, so the bind address is the access control.

## Architecture

`Pipeline` is one object that both the backtester and the live desk run. That is the
design's most important property: the usual way a backtest lies is that simulation and
live trading run different code and diverge somewhere subtle. Here the only difference
is which clock and which broker are injected.

```
data/          feeds: Alpaca spot, Binance perps, synthetic, long history, SQLite cache
analysis/      Murphy: swings, trend, patterns, candles, Elliott, levels, checklist
agents/        each emits a Signal or abstains
alpha/         blends signals into target weights
risk/          money.py = Murphy ch16 sizing; manager.py = account-level veto
execution/     SimBroker (matching), PaperBroker (matching + ledger), Alpaca paper
backtest/      event-driven replay through the live pipeline
psychology/    Douglas: edge statistics, streak probability, discipline monitor
quant/         bootstrap and deflated Sharpe, regimes, Kelly, distributions
ui/            Textual terminals and text charts
web/           stdlib HTTP server + canvas front end
```

Three ways a backtest lies, and what stops each here: fills land at the **next** bar's
open so no decision uses its own outcome; bars from all symbols are merged in strict
timestamp order so nothing peeks cross-sectionally; commission and slippage are charged
on every fill.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

195 tests. The UI tests drive both terminals headlessly through Textual's `run_test()`,
so they pass with no TTY and make no network calls.

## A note on sources

Parameters and quotations are derived from the two books named above. The books
themselves are not in this repository and must not be added to it; `.gitignore`
excludes the local extractions used to source them.
