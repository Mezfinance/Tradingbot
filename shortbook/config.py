"""Settings. Every field can be overridden with an SB_<FIELD> environment variable."""
from __future__ import annotations
import dataclasses, os


@dataclasses.dataclass
class Config:
    db_path: str = "shortbook.db"
    cash_usd: float = 1_000_000.0          # paper fund size
    trade_pct: float = 10.0                # each short = up to this % of NAV
    oi_cap_pct: float = 2.0                # ... and at most this % of the venue's open interest
    gross_cap_pct: float = 100.0           # total open shorts <= this % of NAV
    hold_h: float = 168.0                  # cover after 7 days
    fee_round_trip_pct: float = 0.2        # taker fees + spread, round trip
    impact_k: float = 0.10                 # impact per side = k * sqrt(size / 24h perp volume)
    groups: str = "B:upbit,B:binance,B:coinbase,A:upbit"  # which listing groups are traded
    perp_venues: str = "binance,bybit,gate"
    listing_venues: str = "upbit,coinbase,binance"
    b_min_age_days: float = 7.0            # B: token traded elsewhere at least this long
    a_max_age_days: float = 3.0            # A: token's first listing at most this long ago
    max_price_ratio: float = 2.0           # listing price vs perp price must agree within this factor
    pending_max_h: float = 72.0            # stop waiting for a new market's first trade after this
    max_entry_delay_h: float = 2.0         # skip if we only see trading this long after it began
    detect_s: float = 30.0                 # listing poll interval
    manage_s: float = 300.0                # price / funding / exit pass
    equity_s: float = 3600.0               # equity snapshot interval
    health_port: int = 8080
    watchdog_s: float = 900.0              # /health fails if the loop is older than this

    @classmethod
    def from_env(cls, env=os.environ) -> "Config":
        cfg = cls()
        for f in dataclasses.fields(cls):
            v = env.get("SB_" + f.name.upper())
            if v is None or v == "":
                continue
            t = type(getattr(cfg, f.name))
            setattr(cfg, f.name, t(v))
        return cfg

    def group_set(self) -> set[tuple[str, str]]:
        out = set()
        for g in self.groups.split(","):
            if ":" in g:
                k, v = g.strip().split(":", 1)
                out.add((k.upper(), v.lower()))
        return out

    def names(self, field: str) -> list[str]:
        return [x.strip().lower() for x in getattr(self, field).split(",") if x.strip()]
