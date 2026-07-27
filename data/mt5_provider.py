import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class MT5Provider:
    """
    Provider temps réel via MetaTrader 5 (terminal local).
    Nécessite MT5 installé et connecté à un compte sur le PC.
    """

    def __init__(self, symbol: str = "EURUSD", poll_ms: int = 200):
        self.symbol = symbol
        self.poll_ms = poll_ms
        self._running = False
        self._last_time = 0

    def _init_mt5(self) -> bool:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            logger.error(f"Échec MT5 initialize: {mt5.last_error()}")
            return False
        mt5.symbol_select(self.symbol, True)
        logger.info(f"MT5 connecté, symbole {self.symbol} sélectionné")
        return True

    def _fetch_historical(self, count: int) -> list:
        import MetaTrader5 as mt5
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, count)
        if rates is None or len(rates) == 0:
            logger.error(f"Echec historique MT5: {mt5.last_error()}")
            return []
        candles = []
        for r in rates:
            candles.append({
                "time": datetime.fromtimestamp(r["time"], tz=timezone.utc),
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": r["tick_volume"] or 0,
            })
        logger.info(f"Historique MT5 charge: {len(candles)} bougies")
        return candles

    async def connect(self, candle_builder) -> None:
        import MetaTrader5 as mt5
        if not self._init_mt5():
            return

        # Charge l'historique initial
        hist = await asyncio.to_thread(self._fetch_historical, 500)
        if hist:
            candle_builder.load_candles(hist)

        self._running = True
        logger.info(f"Polling MT5 {self.symbol} toutes les {self.poll_ms}ms...")

        while self._running:
            tick = await asyncio.to_thread(mt5.symbol_info_tick, self.symbol)
            if tick is not None and tick.time != self._last_time:
                self._last_time = tick.time
                mid = (tick.bid + tick.ask) / 2
                ts = datetime.fromtimestamp(tick.time, tz=timezone.utc)
                candle_builder.on_tick(mid, ts)
            await asyncio.sleep(self.poll_ms / 1000)

    async def disconnect(self) -> None:
        self._running = False
        import MetaTrader5 as mt5
        mt5.shutdown()
        logger.info("MT5 déconnecté")
