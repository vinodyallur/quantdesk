# Requirements Document

## Introduction

Three pieces of work were recently completed in QuantDesk and none of them can be inspected interactively: a fix to how per-trade risk is accounted for and reported, a long-history data layer reaching back to 1871, and a perpetual-futures funding and open-interest layer. All three are currently only observable through command-line output or by reading the source.

The gap that matters most is the risk work. A measurement defect allowed reported R-multiple expectancy to be positive while the account lost money. The cause was that 82% of trades were sized by Murphy's 10% market-commitment ceiling rather than the 0.5% risk target, so risk per trade varied across roughly 3,000x and a plain mean of per-trade R was dominated by whichever trades happened to risk least. The fix added a risk-weighted aggregate that cannot disagree in sign with the money, plus dispersion figures that say whether the unweighted number can be trusted at all.

That fix is currently a claim in a document. The purpose of this feature is to make it evidence: a user should be able to see the individual trades with near-zero denominators and watch the two expectancy figures disagree, rather than take a handoff note's word for it.

The existing terminal UI is driven by a live Desk object and polls it on a timer. That is the wrong vehicle here, because long-run history and perps funding are one-shot research pulls with no desk behind them. This feature is a separate, read-only research terminal.

## Glossary

- **R-multiple (R):** A trade's profit or loss expressed in units of the risk it originally took, i.e. `pnl / initial_risk`. Scale-free, so trades of different sizes can be compared.
- **initial_risk:** The currency amount a trade stood to lose if its protective stop were hit, fixed at entry. The denominator of R.
- **mean_r (unweighted expectancy):** The plain average of per-trade R. Valid only when every trade risked a similar amount; otherwise it is dominated by the trades that risked least.
- **risk_weighted_r:** Total P&L divided by total risk taken. Equals `mean_r` when risk per trade is even, and cannot disagree in sign with the P&L because it is the P&L over the risk.
- **risk_dispersion:** Largest per-trade risk divided by the smallest. 1.0 is perfectly even sizing; above roughly 10 the unweighted R figures stop being comparable.
- **Market-commitment ceiling:** Murphy's limit on capital tied up in one market, 10-15% of equity. Distinct from the risk cap, and on an unleveraged account it usually binds first.
- **Risk target:** The intended fraction of equity to lose per trade if stopped out, 0.5% by default. Exists to equalise risk per trade so that R translates into money.
- **Funding rate:** The periodic payment between long and short holders of a perpetual future, settled three times a day on most venues. Positive means longs pay shorts.
- **Annualised carry:** A funding rate extrapolated to a year, expressing the holding cost of a long position as a fraction of notional.
- **Open interest:** Total contracts outstanding. Read together with price, it distinguishes new positions being opened from existing ones being closed.
- **CAPE:** Cyclically adjusted price-to-earnings ratio, from Shiller's dataset. Available from 1881 because it requires ten prior years of earnings.
- **Lookahead:** Using information in a decision that was not available at the time the decision was made. The reason an incomplete trailing period must be excluded from resampled bars.
- **Headless verification:** Driving the UI in tests without a terminal, via Textual's `run_test()` harness.

## Requirements

### Requirement 1: Risk accounting is visible as evidence, not assertion

**User Story:** As a developer who has been told the R-multiple contradiction is fixed, I want to see the underlying per-trade numbers, so that I can confirm the fix rather than trust a summary.

#### Acceptance Criteria

1. WHEN the user runs a backtest from the research terminal THEN the system SHALL display both risk_weighted_r and mean_r adjacent to each other, each labelled to indicate which is trustworthy.
2. WHEN the two expectancy figures differ in sign THEN the system SHALL display an explicit warning stating that the unweighted figure describes the sizing rather than the edge.
3. WHEN backtest results are displayed THEN the system SHALL show median_risk, min_risk, max_risk, and risk_dispersion for risk per trade.
4. WHEN r_is_trustworthy is false THEN the system SHALL visually distinguish the unweighted expectancy figure from the risk-weighted one, so the two are not read as equally valid.
5. WHEN backtest results are displayed THEN the system SHALL present a table of individual trades sorted by absolute R, showing each trade's initial_risk, pnl, peak_qty, and r_multiple.
6. WHEN a trade's risk_known is false THEN the system SHALL mark that trade as excluded from R statistics rather than displaying its R as zero.
7. WHEN the currency expectancy and the risk-weighted expectancy disagree in sign THEN the system SHALL surface this as a defect indicator, because the risk-weighted figure is defined as total P&L over total risk and cannot legitimately disagree.

### Requirement 2: Long history is explorable with its limitations stated

**User Story:** As a researcher considering a long-history model, I want to load and inspect century-scale index data, so that I can judge coverage and quality before fitting anything to it.

#### Acceptance Criteria

1. WHEN the user selects the Yahoo source THEN the system SHALL fetch daily bars and display the bar count, the first and last timestamps, and the span in years.
2. WHEN the user selects the Shiller source THEN the system SHALL display the monthly record count, the coverage window, and the CAPE series start date and latest value.
3. WHEN Shiller data is displayed THEN the system SHALL state that its prices are monthly averages with deliberately equal OHLC values and are not tradeable bars.
4. WHEN loaded bars include any with zero volume THEN the system SHALL report the count and percentage, and state that volume filters cannot operate on those bars.
5. WHEN bars are resampled to months THEN the system SHALL exclude the incomplete trailing month and state that it has been excluded.
6. WHEN a data source is unreachable or returns an error THEN the system SHALL display the failure reason and remain usable.
7. WHEN previously fetched history exists in the local cache THEN the system SHALL load from cache by default and offer an explicit refresh, because the Yahoo endpoint is unofficial and rate limited.
8. WHEN history spans dates before 1970 THEN the system SHALL display those timestamps correctly, since the epoch for such dates is negative.

### Requirement 3: Perps funding and open interest are readable with coverage stated

**User Story:** As a researcher evaluating a perpetual-futures strategy, I want to see funding carry and positioning, so that I can judge whether a strategy's returns survive its holding cost.

#### Acceptance Criteria

1. WHEN the user requests funding history for a symbol THEN the system SHALL display the settlement count, the window covered, the mean and median rate, and the share of settlements where longs paid.
2. WHEN funding history is displayed THEN the system SHALL display the annualised carry for a long position as a percentage.
3. WHEN funding history is displayed THEN the system SHALL state whether the book reads as crowded long, crowded short, or showing no persistent lean.
4. WHEN funding history is displayed THEN the system SHALL list the largest individual settlements with their timestamps and which side paid.
5. WHEN open interest history is available alongside mark prices THEN the system SHALL display the combined reading distinguishing new money from position closing.
6. WHEN open interest history is displayed THEN the system SHALL state the retention limit of the venue queried, because coverage differs materially between venues.
7. WHEN the user selects a venue THEN the system SHALL allow choosing between at least two venues, since divergence between them is itself informative.
8. WHEN a venue does not report mark prices alongside funding THEN the system SHALL state that the price-versus-open-interest reading is unavailable for that venue rather than omitting it silently.

### Requirement 4: The terminal runs without a live desk, keys, or a TTY

**User Story:** As a developer, I want to run the research terminal on a bare checkout, so that inspection does not depend on credentials or a running trading session.

#### Acceptance Criteria

1. WHEN the research terminal starts THEN the system SHALL NOT require a Desk instance.
2. WHEN the research terminal starts THEN the system SHALL NOT require Alpaca API keys.
3. IF the user has no network access THEN the system SHALL still start, and SHALL report per-tab failures rather than failing to launch.
4. WHEN the terminal is launched THEN the system SHALL be reachable as a subcommand of the existing quantdesk CLI, consistent with the existing commands.

### Requirement 5: Slow work never blocks the interface

**User Story:** As a user, I want the terminal to stay responsive while data loads, so that I can tell the difference between working and hung.

#### Acceptance Criteria

1. WHEN a backtest or network fetch is triggered THEN the system SHALL execute it off the UI thread.
2. WHILE a long-running operation is in progress THE system SHALL display an in-progress indication naming the operation.
3. WHILE a long-running operation is in progress THE system SHALL keep navigation between tabs responsive.
4. IF a long-running operation is already in progress THEN the system SHALL prevent a second concurrent run of that same operation.
5. WHEN an operation completes THEN the system SHALL replace the in-progress indication with the result or the failure reason.

### Requirement 6: The terminal is verifiable headlessly

**User Story:** As a developer working in an environment with no interactive terminal, I want automated verification of the UI, so that it is tested rather than assumed to work.

#### Acceptance Criteria

1. WHEN the test suite runs THEN the system SHALL verify the research terminal mounts successfully without a TTY.
2. WHEN the test suite runs THEN the system SHALL verify that each tab renders its expected content.
3. WHEN the test suite runs THEN the system SHALL verify the risk panel against synthetic results with known values, including a case where the unweighted and risk-weighted figures disagree in sign.
4. WHEN the test suite runs THEN the system SHALL NOT make network calls, so that tests are deterministic and pass offline.
5. IF a panel raises an exception during refresh THEN the system SHALL log the failure and continue rendering the remaining panels.

### Requirement 7: Visual consistency with the existing terminal

**User Story:** As a user of the existing live terminal, I want the research terminal to read the same way, so that I am not learning a second visual language.

#### Acceptance Criteria

1. WHEN any panel is rendered THEN the system SHALL use the existing amber-on-black palette.
2. WHEN a signed number is displayed THEN the system SHALL colour it green for positive, red for negative, and neutral for zero, matching existing conventions.
3. WHEN the terminal is displayed THEN the system SHALL show key bindings for available actions.
4. WHEN panels display tabular data THEN the system SHALL use the same table styling as the existing terminal.
