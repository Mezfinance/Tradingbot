# Short-only fund: strategies

Research only. Nothing here trades. Scripts and the event table are in this folder; working data (cex/, liquid/) lived in the session scratchpad.

## Strategy 1: CEX listing shorts

Thesis: when a major exchange lists a token, the price has usually run up before the listing and declines in the following days. Short at the listing hour, cover after about 7 days.

Evidence (short from the listing hour, 7-day hold, 0.2% round trip, net of Binance funding where available; see CLAUDE.md 2026-11-02 entries):

| Group | n | Mean per trade | 90% CI | Win rate |
|---|---|---|---|---|
| B Binance (token already on another venue 7+ days) | 34 | +20.3% | +14.3..+26.6 | ~80% |
| A Upbit (token's first CEX listings) | 17 | +24.4% | | ~95% |
| B Upbit | 85 | +12.4% | +8.6..+15.9 | ~85% |
| B Coinbase | 50 | +5.2% | +0.5..+9.6 | ~66% |

- Venues: Binance + Bybit main (86-91% of events shortable at the listing hour with two venues), Gate third. MEXC caps new contracts at ~$5k. Hyperliquid covers only ~40%.
- Capacity: Binance open interest in the first 48 h has a median of $16M (p10 $5.5M).
- Entry timing: the listing hour beats the announcement (see below).
- Fund manager is in Australia: venue access and fund structure need Australian legal/AFSL advice.

### Announcement vs listing entry (2026-11-02, `ann_binance.py`, `ann_analyse.py`)

Binance's 'New Cryptocurrency Listing' notices (catalog 48) matched to 174 of 237 Binance listing events.
- Lead time from announcement to listing: B median 4.3 h (p10 0.5, p90 9.1); A median 9.0 h (p90 89 h).
- B Binance (n=42 with prices): the run-up happens **before** the announcement (7 days before -> announcement +58.6% mean, +17.8% median). From announcement to listing the price is flat (+0.8%, median -1.6%).
- Shorting at the announcement does worse than at the listing: 7-day exit +17.2% (CI +5.1..+26.9) vs +19.4% (+12.5..+25.7). The hour after the announcement squeezes shorts (-3.4%, win 39%).
- A events can't be shorted at the announcement: the token has no market until it lists.
- Upbit Korea's notice API is blocked from the research sandbox (Cloudflare), so Upbit announcement timing is untested.

Next: per-period stability of the venue-shortable sets; a live detector for new Upbit KRW / Coinbase / Binance listings; a paper short book with real perp prices, funding and position limits.

## Strategy 2+

The co-founder wants more short signals beyond the listing thesis. To be specified.
