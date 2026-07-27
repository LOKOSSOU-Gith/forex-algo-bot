import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from dukascopy_python import fetch, INTERVAL_MIN_1, OFFER_SIDE_BID

from data.ws_providers import CandleBuilder

logger = logging.getLogger(__name__)

# Offre-side par défaut (on utilise BID pour correspondre au prix spot)
_DEFAULT_OFFER = OFFER_SIDE_BID


class DukascopyPollProvider:
    """Provider REST polling via Dukascopy.

    Interroge Dukascopy toutes les ~60s pour récupérer les dernières
    bougies 1-minute d'EUR/USD et les injecte dans le CandleBuilder.

    Dukascopy fournit des données Forex gratuites de haute qualité
    (tick et 1-minute) sans clé API ni inscription.
    """

    def __init__(
        self,
        instrument: str = "EUR/USD",
        poll_interval: int = 60,
        window_minutes: int = 15,
    ):
        self.instrument = instrument
        self.poll_interval = poll_interval
        self.window_minutes = window_minutes
        self._running = False

    async def _fetch_candles(self) -> pd.DataFrame:
        """Récupère les dernières bougies Dukascopy (async, thread séparé)."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=self.window_minutes)

        df = await asyncio.to_thread(
            fetch,
            self.instrument,
            INTERVAL_MIN_1,
            _DEFAULT_OFFER,
            start,
            end,
            limit=100,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        return df

    def _to_candle(self, ts, row: pd.Series) -> dict:
        """Convertit une ligne Dukascopy en dict bougie standard."""
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        return {
            "time": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    async def connect(self, candle_builder: CandleBuilder) -> None:
        self._running = True
        logger.info(
            f"Démarrage polling Dukascopy ({self.instrument}) "
            f"toutes les {self.poll_interval}s"
        )

        while self._running:
            try:
                df = await self._fetch_candles()

                if df.empty:
                    await asyncio.sleep(self.poll_interval)
                    continue

                nouvelles = 0
                for ts, row in df.iterrows():
                    try:
                        candle = self._to_candle(ts, row)
                        avant = len(candle_builder._candles)
                        candle_builder.add_complete_candle(candle)
                        if len(candle_builder._candles) > avant:
                            nouvelles += 1
                    except Exception as e:
                        logger.warning(
                            f"Erreur ajout bougie Dukascopy: {e}"
                        )
                        continue

                if nouvelles:
                    logger.info(
                        f"Poll Dukascopy: {nouvelles} nouvelle(s) bougie(s)"
                    )

            except asyncio.CancelledError:
                logger.info("Polling Dukascopy annulé")
                break
            except Exception as e:
                logger.warning(f"Erreur poll Dukascopy: {e}")

            await asyncio.sleep(self.poll_interval)

    async def disconnect(self) -> None:
        self._running = False
        logger.info("Polling Dukascopy arrêté")


async def fetch_history(
    instrument: str = "EUR/USD",
    count: int = 500,
) -> pd.DataFrame:
    """Récupère l'historique des bougies 1-minute pour le chargement initial.

    Retourne un DataFrame avec le timestamp en index (timezone-naive)
    et colonnes: open, high, low, close, volume.
    Format compatible avec CandleBuilder.load_history().
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=count + 60)

    df = await asyncio.to_thread(
        fetch,
        instrument,
        INTERVAL_MIN_1,
        _DEFAULT_OFFER,
        start,
        end,
        limit=count + 60,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Garder seulement les count dernières bougies
    df = df.tail(count).copy()

    # Rendre l'index timestamp timezone-naive (compatible load_history)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df
