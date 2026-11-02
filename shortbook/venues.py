"""Exchange clients (public endpoints only, no keys). All fail soft: None / {} / [] when a venue doesn't answer.

Listing venues (where a new spot market appears): Upbit KRW, Coinbase USD/USDC, Binance spot.
Perp venues (where we short): Binance USD-M, Bybit linear, Gate USDT futures.
Age sources (when a token first traded anywhere): perp launch times, Binance spot first candle, Gate and OKX
spot listing times."""
from __future__ import annotations
import dataclasses, datetime as dt, time
from .http import Http

STABLE = {"USDT", "USDC", "USD", "FDUSD", "TUSD", "DAI", "USDE", "PYUSD", "EUR", "EURC", "USD1", "RLUSD", "USDD",
          "BUSD", "WBTC", "WETH", "STETH", "WSTETH", "CBBTC", "BTC", "ETH", "KRW", "XUSD", "USDS"}
PREFIXES = (("1000000", 1_000_000), ("100000", 100_000), ("10000", 10_000), ("1000", 1000), ("1M", 1_000_000))


def norm(base: str) -> tuple[str, float]:
    """Perp base -> (token base, units per contract unit): '1000PEPE' -> ('PEPE', 1000)."""
    b = base.upper()
    for p, m in PREFIXES:
        if b.startswith(p) and len(b) > len(p) + 1:
            return b[len(p):], float(m)
    return b, 1.0


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _utc(s: str) -> float:
    return dt.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()


@dataclasses.dataclass
class Contract:
    venue: str
    symbol: str
    base: str           # token base after removing a 1000-style prefix
    mult: float         # token units per quoted unit (price / mult = token price)
    launch_ts: float | None


@dataclasses.dataclass
class Quote:
    price: float        # last price per token (USD)
    mark: float         # mark price per token
    oi_usd: float | None
    vol24_usd: float | None
    funding_rate: float | None


# ---------------------------------------------------------------- listing venues
class Upbit:
    name = "upbit"
    B = "https://api.upbit.com/v1"

    def __init__(self, http: Http):
        self.h = http

    def markets(self) -> dict[str, str] | None:
        d = self.h.get(f"{self.B}/market/all")
        if not isinstance(d, list):
            return None
        return {m["market"][4:]: m["market"] for m in d if m.get("market", "").startswith("KRW-")}

    def first_trade(self, symbol: str) -> float | None:
        d = self.h.get(f"{self.B}/candles/minutes/1", {"market": symbol, "count": 200})
        if not isinstance(d, list) or not d:
            return None
        return min(_utc(c["candle_date_time_utc"]) for c in d)

    def price_usd(self, symbol: str) -> float | None:
        d = self.h.get(f"{self.B}/ticker", {"markets": f"{symbol},KRW-USDT"})
        if not isinstance(d, list):
            return None
        px = {x["market"]: _f(x.get("trade_price")) for x in d}
        if px.get(symbol) and px.get("KRW-USDT"):
            return px[symbol] / px["KRW-USDT"]
        return None


class Coinbase:
    name = "coinbase"
    B = "https://api.exchange.coinbase.com"

    def __init__(self, http: Http):
        self.h = http

    def markets(self) -> dict[str, str] | None:
        d = self.h.get(f"{self.B}/products")
        if not isinstance(d, list):
            return None
        out: dict[str, str] = {}
        for p in sorted(d, key=lambda p: p.get("quote_currency") != "USD"):   # prefer USD over USDC
            if p.get("quote_currency") in ("USD", "USDC"):
                out.setdefault(p["base_currency"].upper(), p["id"])
        return out

    def first_trade(self, symbol: str) -> float | None:
        d = self.h.get(f"{self.B}/products/{symbol}/candles", {"granularity": 60})
        if not isinstance(d, list) or not d:
            return None
        return float(min(c[0] for c in d))

    def price_usd(self, symbol: str) -> float | None:
        d = self.h.get(f"{self.B}/products/{symbol}/ticker")
        return _f(d.get("price")) if isinstance(d, dict) else None


class BinanceSpot:
    """Public market data mirror (data-api.binance.vision), reachable where api.binance.com is not.
    The full exchangeInfo (~MBs) is read once per `full_every_s`; in between, the much smaller price list shows new
    symbols, which are then looked up one by one."""
    name = "binance"
    B = "https://data-api.binance.vision/api/v3"

    def __init__(self, http: Http, full_every_s: float = 6 * 3600):
        self.h, self.full_every_s = http, full_every_s
        self._sym_base: dict[str, str] = {}
        self._full_ts = 0.0

    def _full(self) -> bool:
        d = self.h.get(f"{self.B}/exchangeInfo")
        if not isinstance(d, dict) or "symbols" not in d:
            return False
        self._sym_base = {s["symbol"]: s["baseAsset"].upper() for s in d["symbols"]}
        self._full_ts = time.time()
        return True

    def markets(self) -> dict[str, str] | None:
        if (not self._sym_base or time.time() - self._full_ts > self.full_every_s) and not self._full():
            return None
        prices = self.h.get(f"{self.B}/ticker/price")
        if isinstance(prices, list):
            for p in prices:
                s = p.get("symbol")
                if s and s not in self._sym_base:
                    d = self.h.get(f"{self.B}/exchangeInfo", {"symbol": s})
                    if isinstance(d, dict) and d.get("symbols"):
                        self._sym_base[s] = d["symbols"][0]["baseAsset"].upper()
        out: dict[str, str] = {}
        for s, b in sorted(self._sym_base.items(), key=lambda kv: not kv[0].endswith("USDT")):  # prefer USDT
            out.setdefault(b, s)
        return out

    def first_trade(self, symbol: str) -> float | None:
        d = self.h.get(f"{self.B}/klines", {"symbol": symbol, "interval": "1m", "startTime": 0, "limit": 1})
        return d[0][0] / 1000 if isinstance(d, list) and d else None

    def first_trade_base(self, base: str) -> float | None:
        """Age evidence: first candle of BASEUSDT (None if Binance has no such market)."""
        return self.first_trade(f"{base}USDT")

    def price_usd(self, symbol: str) -> float | None:
        d = self.h.get(f"{self.B}/ticker/price", {"symbol": symbol})
        if not isinstance(d, dict):
            return None
        px = _f(d.get("price"))
        return px if symbol.endswith(("USDT", "USDC", "FDUSD")) else None


# ---------------------------------------------------------------- perp venues
class BinancePerp:
    name = "binance"
    B = "https://fapi.binance.com/fapi/v1"

    def __init__(self, http: Http):
        self.h = http

    def contracts(self) -> dict[str, Contract] | None:
        d = self.h.get(f"{self.B}/exchangeInfo")
        if not isinstance(d, dict) or "symbols" not in d:
            return None
        out = {}
        for s in d["symbols"]:
            if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
                continue
            base, mult = norm(s["baseAsset"])
            out[base] = Contract(self.name, s["symbol"], base, mult, (s.get("onboardDate") or 0) / 1000 or None)
        return out

    def quote(self, c: Contract) -> Quote | None:
        pi = self.h.get(f"{self.B}/premiumIndex", {"symbol": c.symbol})
        tk = self.h.get(f"{self.B}/ticker/24hr", {"symbol": c.symbol})
        oi = self.h.get(f"{self.B}/openInterest", {"symbol": c.symbol})
        if not isinstance(pi, dict) or not isinstance(tk, dict):
            return None
        mark = _f(pi.get("markPrice"))
        last = _f(tk.get("lastPrice")) or mark
        if not last:
            return None
        oi_units = _f(oi.get("openInterest")) if isinstance(oi, dict) else None
        return Quote(last / c.mult, (mark or last) / c.mult, oi_units * (mark or last) if oi_units is not None else None,
                     _f(tk.get("quoteVolume")), _f(pi.get("lastFundingRate")))

    def funding(self, c: Contract, since_ts: float) -> list[tuple[float, float]]:
        d = self.h.get(f"{self.B}/fundingRate", {"symbol": c.symbol, "startTime": int(since_ts * 1000), "limit": 1000})
        if not isinstance(d, list):
            return []
        return [(x["fundingTime"] / 1000, float(x["fundingRate"])) for x in d]

    def limits(self, c: Contract) -> dict:
        return {}   # Binance's leverage brackets need an API key


class BybitPerp:
    name = "bybit"
    B = "https://api.bybit.com/v5/market"

    def __init__(self, http: Http):
        self.h = http

    def contracts(self) -> dict[str, Contract] | None:
        out, cursor = {}, None
        for _ in range(20):
            p = {"category": "linear", "limit": 1000, **({"cursor": cursor} if cursor else {})}
            d = self.h.get(f"{self.B}/instruments-info", p)
            if not isinstance(d, dict) or d.get("retCode") != 0:
                return None if not out else out
            r = d.get("result") or {}
            for s in r.get("list") or []:
                if s.get("contractType") != "LinearPerpetual" or s.get("quoteCoin") != "USDT" or s.get("status") != "Trading":
                    continue
                base, mult = norm(s["baseCoin"])
                out[base] = Contract(self.name, s["symbol"], base, mult, (_f(s.get("launchTime"), 0) / 1000) or None)
            cursor = r.get("nextPageCursor")
            if not cursor:
                break
        return out

    def quote(self, c: Contract) -> Quote | None:
        d = self.h.get(f"{self.B}/tickers", {"category": "linear", "symbol": c.symbol})
        try:
            t = d["result"]["list"][0]
        except (TypeError, KeyError, IndexError):
            return None
        last, mark = _f(t.get("lastPrice")), _f(t.get("markPrice"))
        if not last:
            return None
        return Quote(last / c.mult, (mark or last) / c.mult, _f(t.get("openInterestValue")), _f(t.get("turnover24h")),
                     _f(t.get("fundingRate")))

    def funding(self, c: Contract, since_ts: float) -> list[tuple[float, float]]:
        """Bybit wants startTime and endTime together; without them it returns the latest 200 settlements.
        Ask with both, and fall back to the latest 200 (filtered) if that answer is an error or empty."""
        base = {"category": "linear", "symbol": c.symbol, "limit": 200}
        rows = self._rows(self.h.get(f"{self.B}/funding/history",
                                     {**base, "startTime": int(since_ts * 1000), "endTime": int(time.time() * 1000)}))
        if not rows:
            rows = self._rows(self.h.get(f"{self.B}/funding/history", base))
        out = sorted((int(x["fundingRateTimestamp"]) / 1000, float(x["fundingRate"])) for x in rows)
        return [(t, r) for t, r in out if t >= since_ts]

    @staticmethod
    def _rows(d) -> list:
        try:
            return list(d["result"]["list"]) if d.get("retCode") == 0 else []
        except (AttributeError, TypeError, KeyError):
            return []

    def limits(self, c: Contract) -> dict:
        d = self.h.get(f"{self.B}/risk-limit", {"category": "linear", "symbol": c.symbol})
        try:
            tiers = [_f(x.get("riskLimitValue")) for x in d["result"]["list"]]
        except (TypeError, KeyError):
            return {}
        tiers = [t for t in tiers if t]
        return {"first_tier_usd": min(tiers), "max_usd": max(tiers)} if tiers else {}


class GatePerp:
    name = "gate"
    B = "https://api.gateio.ws/api/v4/futures/usdt"

    def __init__(self, http: Http):
        self.h = http
        self._qm: dict[str, float] = {}

    def contracts(self) -> dict[str, Contract] | None:
        d = self.h.get(f"{self.B}/contracts")
        if not isinstance(d, list):
            return None
        out = {}
        for s in d:
            if s.get("in_delisting") or not s.get("name", "").endswith("_USDT"):
                continue
            base, mult = norm(s["name"][:-5])
            self._qm[s["name"]] = _f(s.get("quanto_multiplier"), 1.0)
            out[base] = Contract(self.name, s["name"], base, mult, _f(s.get("launch_time")) or _f(s.get("create_time")))
        return out

    def quote(self, c: Contract) -> Quote | None:
        d = self.h.get(f"{self.B}/tickers", {"contract": c.symbol})
        if not isinstance(d, list) or not d:
            return None
        t = d[0]
        last, mark = _f(t.get("last")), _f(t.get("mark_price"))
        if not last:
            return None
        size = _f(t.get("total_size"))
        qm = self._qm.get(c.symbol, 1.0)
        oi = size * qm * (mark or last) if size is not None else None
        return Quote(last / c.mult, (mark or last) / c.mult, oi, _f(t.get("volume_24h_quote")), _f(t.get("funding_rate")))

    def funding(self, c: Contract, since_ts: float) -> list[tuple[float, float]]:
        d = self.h.get(f"{self.B}/funding_rate", {"contract": c.symbol, "from": int(since_ts), "limit": 1000})
        if not isinstance(d, list):
            return []
        return sorted((float(x["t"]), float(x["r"])) for x in d if float(x["t"]) >= since_ts)

    def limits(self, c: Contract) -> dict:
        d = self.h.get(f"{self.B}/contracts/{c.symbol}")
        if not isinstance(d, dict):
            return {}
        return {"risk_limit_base_usd": _f(d.get("risk_limit_base")), "risk_limit_max_usd": _f(d.get("risk_limit_max"))}


# ---------------------------------------------------------------- age sources
class SpotAges:
    """First listing time per base on Gate and OKX spot (one call each, cached for an hour)."""

    def __init__(self, http: Http, ttl_s: float = 3600):
        self.h, self.ttl_s = http, ttl_s
        self._ages: dict[str, dict[str, float]] = {}
        self._ts = 0.0

    def ages(self) -> dict[str, dict[str, float]]:
        if self._ages and time.time() - self._ts < self.ttl_s:
            return self._ages
        out: dict[str, dict[str, float]] = {}
        d = self.h.get("https://api.gateio.ws/api/v4/spot/currency_pairs")
        if isinstance(d, list):
            for p in d:
                t = _f(p.get("buy_start"))
                if t and p.get("quote") in ("USDT", "USDC"):
                    b = p["base"].upper()
                    out.setdefault(b, {})["gate"] = min(t, out.get(b, {}).get("gate", t))
        d = self.h.get("https://www.okx.com/api/v5/public/instruments", {"instType": "SPOT"})
        if isinstance(d, dict):
            for p in d.get("data") or []:
                t = _f(p.get("listTime"))
                if t:
                    b = p["baseCcy"].upper()
                    out.setdefault(b, {})["okx"] = min(t / 1000, out.get(b, {}).get("okx", t / 1000))
        if out:
            self._ages, self._ts = out, time.time()
        return self._ages


LISTING = {"upbit": Upbit, "coinbase": Coinbase, "binance": BinanceSpot}
PERPS = {"binance": BinancePerp, "bybit": BybitPerp, "gate": GatePerp}
