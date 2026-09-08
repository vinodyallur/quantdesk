"""Central configuration, loaded from environment / .env file.

Every setting is overridable with a ``QD_`` prefixed environment variable, e.g.
``QD_TIMEFRAME=5Min``. Secrets are never logged or printed by the app.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AssetClass = Literal["crypto", "us_equity"]
Timeframe = Literal["1Min", "5Min", "15Min", "1Hour", "1Day"]


class Settings(BaseSettings):
    """Runtime configuration for the desk."""

    model_config = SettingsConfigDict(
        env_prefix="QD_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ---------------------------------------------------------------- broker
    alpaca_api_key: SecretStr | None = None
    alpaca_secret_key: SecretStr | None = None
    # Hard-wired to paper. Flipping this to False is intentionally not enough to
    # trade real money: execution/alpaca_broker.py also refuses live endpoints.
    paper: bool = True

    # ------------------------------------------------------------------ data
    # Comma separated. Crypto is the default because Alpaca serves crypto bars
    # without any API key, so the desk runs end-to-end with zero setup.
    universe_raw: str = Field(
        default="BTC/USD,ETH/USD,LTC/USD,BCH/USD",
        alias="QD_UNIVERSE",
    )
    asset_class: AssetClass = "crypto"
    timeframe: Timeframe = "5Min"
    # Stock data feed: "iex" is free, "sip" needs a paid subscription.
    stock_feed: Literal["iex", "sip"] = "iex"
    # How many historical bars to warm the feature engine with on startup.
    warmup_bars: int = 300
    # Live loop poll interval in seconds (used when websocket keys are absent).
    poll_seconds: int = 20

    # --------------------------------------------------------------- capital
    starting_equity: float = 100_000.0

    # ------------------------------------------------------------------ risk
    #: Max absolute weight of equity in any single symbol.
    max_position_weight: float = 0.20
    #: Max sum of |weight| across all positions (leverage ceiling).
    max_gross_exposure: float = 1.00
    #: Max |sum of signed weights| (directional tilt ceiling).
    max_net_exposure: float = 0.60
    #: Stop opening new risk after losing this fraction of start-of-day equity.
    daily_loss_limit_pct: float = 0.03
    #: Flatten everything and halt if drawdown from peak equity exceeds this.
    max_drawdown_pct: float = 0.15
    #: Blended signal strength required before the desk will act at all.
    min_signal_score: float = 0.15
    #: Bars to wait before re-trading a symbol after a fill.
    cooldown_bars: int = 3
    #: Don't bother rebalancing unless target differs from current by this much.
    rebalance_threshold: float = 0.02
    #: Target annualised volatility used to size positions.
    target_volatility: float = 0.20

    # ----------------------------------------------------------------- costs
    commission_bps: float = 1.0
    slippage_bps: float = 2.0

    # ---------------------------------------------------------------- agents
    #: The Murphy analysis agents (checklist, patterns, structure, divergence,
    #: candles, Elliott). They carry the heaviest weights in the roster, so turning
    #: them off leaves a purely statistical desk.
    enable_murphy_agents: bool = True
    #: Enforce Murphy's "work from the long term to the short term" as a hard veto:
    #: no trade may oppose the higher-timeframe direction.
    require_timeframe_agreement: bool = True
    #: Murphy's chapter 16 reward-to-risk yardstick. Trades below this are refused
    #: outright rather than sized down.
    min_reward_risk: float = 3.0
    #: Fraction of equity that may be lost on one market if its stop is hit. Murphy's
    #: hard ceiling.
    max_risk_per_trade_pct: float = 0.05
    #: Fraction of equity to actually risk per trade. Sizing against the ceiling alone
    #: lets risk vary by 30x between trades depending on which limit binds, which stops
    #: a positive expectancy in R from compounding into money. See risk/money.py.
    risk_target_pct: float = 0.005
    #: Account leverage. 1.0 means cash trading, which is what an Alpaca crypto
    #: paper account is; see risk/money.py for why this changes which limit binds.
    leverage: float = 1.0
    #: Minimum protective-stop distance in ATR. A stop closer than the market's own
    #: bar-to-bar range gets taken out by noise regardless of whether the idea was
    #: right, so structural stops are widened to this floor and position size shrinks
    #: to keep the risk budget constant.
    min_stop_atr: float = 2.0
    #: Bars a trade may be held before a time stop closes it. 0 disables it.
    max_bars_held: int = 0

    enable_llm_agent: bool = False
    llm_model: str = "gpt-4o-mini"
    openai_api_key: SecretStr | None = None
    #: Seconds between LLM research passes. Deliberately slow and off hot path.
    llm_interval_seconds: int = 900

    # --------------------------------------------------------------- storage
    data_dir: Path = PROJECT_ROOT / "data"
    log_level: str = "INFO"

    # ------------------------------------------------------------- validators
    @field_validator("data_dir", mode="after")
    @classmethod
    def _ensure_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    # ---------------------------------------------------------------- helpers
    @property
    def universe(self) -> list[str]:
        """Universe as a clean list of symbols."""
        return [s.strip().upper() for s in self.universe_raw.split(",") if s.strip()]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "quantdesk.sqlite"

    @property
    def has_alpaca_keys(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    @property
    def is_crypto(self) -> bool:
        return self.asset_class == "crypto"

    def key(self) -> str | None:
        return self.alpaca_api_key.get_secret_value() if self.alpaca_api_key else None

    def secret(self) -> str | None:
        return (
            self.alpaca_secret_key.get_secret_value() if self.alpaca_secret_key else None
        )

    def openai_key(self) -> str | None:
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None

    @property
    def timeframe_minutes(self) -> int:
        """Bar length in minutes, used for annualisation math."""
        return {
            "1Min": 1,
            "5Min": 5,
            "15Min": 15,
            "1Hour": 60,
            "1Day": 1440,
        }[self.timeframe]

    @property
    def bars_per_year(self) -> float:
        """Approximate bars per year, for Sharpe / vol annualisation."""
        if self.is_crypto:
            minutes_per_year = 365 * 24 * 60  # crypto trades continuously
        else:
            minutes_per_year = 252 * 6.5 * 60  # US equity regular session
        return minutes_per_year / self.timeframe_minutes


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
