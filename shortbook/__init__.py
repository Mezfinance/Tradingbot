"""Paper short book for Strategy 1 of the short-only fund (research/listings/STRATEGIES.md).

Detects new spot listings on Upbit (KRW), Coinbase and Binance, classifies each as A (a token's first
exchange listings) or B (a token already trading elsewhere for 7+ days), and paper-shorts the perpetual on
Binance, Bybit or Gate at the listing hour, covering 7 days later. Never places real orders."""
