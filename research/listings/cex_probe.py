import httpx, json
c = httpx.Client(timeout=20, verify="/root/.ccr/ca-bundle.crt", follow_redirects=True)
probes = {
 "binance_vision_klines": "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1",
 "binance_fapi_exchangeinfo": "https://fapi.binance.com/fapi/v1/exchangeInfo",
 "bybit_spot_instruments": "https://api.bybit.com/v5/market/instruments-info?category=spot&limit=5",
 "bybit_linear_instruments": "https://api.bybit.com/v5/market/instruments-info?category=linear&limit=5",
 "bybit_kline": "https://api.bybit.com/v5/market/kline?category=spot&symbol=BTCUSDT&interval=D&limit=1",
 "okx_instruments_spot": "https://www.okx.com/api/v5/public/instruments?instType=SPOT",
 "okx_instruments_swap": "https://www.okx.com/api/v5/public/instruments?instType=SWAP",
 "okx_history_candles": "https://www.okx.com/api/v5/market/history-candles?instId=BTC-USDT&bar=1D&limit=1",
 "coinbase_products": "https://api.exchange.coinbase.com/products",
 "coinbase_candles": "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400",
 "kraken_assetpairs": "https://api.kraken.com/0/public/AssetPairs",
 "kucoin_symbols": "https://api.kucoin.com/api/v2/symbols",
 "kucoin_klines": "https://api.kucoin.com/api/v1/market/candles?type=1day&symbol=BTC-USDT",
 "gate_pairs": "https://api.gateio.ws/api/v4/spot/currency_pairs",
 "gate_candles": "https://api.gateio.ws/api/v4/spot/candlesticks?currency_pair=BTC_USDT&interval=1d&limit=1",
 "mexc_exchangeinfo": "https://api.mexc.com/api/v3/exchangeInfo",
 "bitget_symbols": "https://api.bitget.com/api/v2/spot/public/symbols",
 "htx_symbols": "https://api.huobi.pro/v2/settings/common/symbols",
 "upbit_markets": "https://api.upbit.com/v1/market/all",
 "upbit_candles": "https://api.upbit.com/v1/candles/days?market=KRW-BTC&count=1",
 "bithumb_ticker": "https://api.bithumb.com/public/ticker/ALL_KRW",
 "hyperliquid_meta": "https://api.hyperliquid.xyz/info",
 "binance_announcements": "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&catalogId=48&pageNo=1&pageSize=5",
 "coingecko": "https://api.coingecko.com/api/v3/ping",
}
for k, u in probes.items():
    try:
        r = c.post(u, json={"type": "meta"}) if k == "hyperliquid_meta" else c.get(u)
        print(f"{k:28} {r.status_code} {len(r.content):>9}B {r.text[:90]!r}")
    except Exception as e:
        print(f"{k:28} ERR {type(e).__name__}: {str(e)[:80]}")
