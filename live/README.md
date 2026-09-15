# Live strategy skeleton

Current intended loop:

1. During TSE sessions, scan 5-minute candles.
2. Previous candle must be bearish (`close < open`); bullish/doji is rejected.
3. Compute previous candle midpoint `(high + low) / 2`.
4. When the current candle touches/crosses that midpoint, create a buy signal.
5. Hold only one position at a time.
6. Gross profit target is configurable in `Config.target_profit_yen`; fees/taxes are intentionally excluded.
7. At the target, call `sell_all`, clear the position, and immediately resume scanning.
8. Continue throughout TSE trading sessions.

## Safety boundary

The strategy and exit loop are implemented, but the broker adapter deliberately has no credentials and no real-order transport. Real Japanese-stock order submission must only be connected after an officially supported broker order interface is confirmed and tested in a non-live environment.

## Important execution detail

A 5-minute OHLC bar can show that the midpoint was touched but cannot prove the intrabar sequence or a real fill. Live use therefore needs a real-time quote/trade feed and broker acknowledgement before a position is considered open.
