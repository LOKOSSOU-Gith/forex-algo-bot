import asyncio
import logging
import os

import pandas as pd

# Désactiver TOUS les warnings yfinance (numpy/pandas internes)
os.environ["PYTHONWARNINGS"] = "ignore"

from data.provider import YFinanceProvider
from data.ws_providers import CandleBuilder

logger = logging.getLogger(__name__)


class YFinancePollProvider:
    """Provider REST polling via yfinance.

    Interroge yfinance toutes les ~60 secondes pour récupérer les
    dernières bougies 1-minute et les injecte dans le CandleBuilder.

    Attention : yfinance diffuse les données avec ~15 min de retard
    (compte gratuit). Ce provider est utile pour tester la pipeline
    complète du bot (stratégie, ML, Discord) sans flux temps réel.
    """

    def __init__(
        self,
        symbol: str = "EURUSD",
        poll_interval: int = 120,  # 2 min entre chaque poll (limite rate-limit)
        candles_per_poll: int = 5,
    ):
        self.symbol = symbol
        self.poll_interval = poll_interval
        self.candles_per_poll = candles_per_poll
        self._running = False
        self._yf = YFinanceProvider()

    def _to_candle(self, row: pd.Series) -> dict:
        ts = row["time"]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        return {
            "time": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
        }

    async def connect(self, candle_builder: CandleBuilder) -> None:
        self._running = True
        logger.info(
            f"Démarrage polling yfinance ({self.symbol}) "
            f"toutes les {self.poll_interval}s"
        )

        while self._running:
            try:
                df = self._yf.fetch_rates(self.symbol, self.candles_per_poll)
                if df.empty:
                    logger.info("Poll yfinance: pas de nouvelles données (rate-limit ou délai ~15 min)")
                    await asyncio.sleep(self.poll_interval)
                    continue

                for _, row in df.iterrows():
                    try:
                        candle = self._to_candle(row)
                        avant = len(candle_builder._candles)
                        candle_builder.add_complete_candle(candle)
                        if len(candle_builder._candles) > avant:
                            logger.debug(
                                f"Nouvelle bougie: {candle['time']} "
                                f"close={candle['close']}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Erreur ajout bougie yfinance: {e}"
                        )
                        continue

            except asyncio.CancelledError:
                logger.info("Polling yfinance annulé")
                break
            except Exception as e:
                logger.warning(f"Erreur poll yfinance: {e}")

            await asyncio.sleep(self.poll_interval)

    async def disconnect(self) -> None:
        self._running = False
        logger.info("Polling yfinance arrêté")
