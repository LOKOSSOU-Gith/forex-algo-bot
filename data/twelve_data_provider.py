import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class TwelveDataWSProvider:
    """
    Provider temps réel via Twelve Data WebSocket.
    Diffuse les ticks EUR/USD en temps réel.
    
    Documentation: https://twelvedata.com/docs/websockets
    URL: wss://ws.twelvedata.com/v1/quotes/price?apikey=...
    """

    def __init__(self, api_key: str, symbol: str = "EUR/USD"):
        self.api_key = api_key
        self.symbol = symbol
        self._ws = None
        self._running = False

    def _ws_url(self) -> str:
        return f"wss://ws.twelvedata.com/v1/quotes/price?apikey={self.api_key}"

    async def connect(self, candle_builder) -> None:
        import websockets

        retry_delay = 5  # secondes entre chaque tentative de reconnexion

        while True:
            try:
                uri = self._ws_url()
                logger.info(f"Connexion Twelve Data WebSocket: {uri[:60]}...")

                async with websockets.connect(uri) as ws:
                    self._ws = ws
                    self._running = True

                    # Abonnement à la paire
                    sub = {
                        "action": "subscribe",
                        "params": {"symbols": self.symbol},
                    }
                    await ws.send(json.dumps(sub))
                    logger.info(f"Abonné à {self.symbol} sur Twelve Data")

                    # Boucle de réception des ticks
                    async for raw in ws:
                        if not raw.strip():
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        # Message heartbeat / statut
                        if msg.get("event") == "subscribe-status":
                            logger.info(f"Twelve Data subscribe-status: {msg.get('status', 'ok')}")
                            continue

                        # Message price tick
                        if msg.get("event") == "price":
                            try:
                                price = float(msg.get("price", 0))
                                if price == 0:
                                    continue
                                ts_raw = msg.get("timestamp")
                                if ts_raw:
                                    ts = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
                                else:
                                    ts = datetime.now(timezone.utc)
                                candle_builder.on_tick(price, ts)
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Erreur parsing tick Twelve Data: {e} {msg}")
                                continue

            except websockets.exceptions.WebSocketException as e:
                logger.warning(f"WebSocket déconnecté ({e}). Reconnexion dans {retry_delay}s...")
            except asyncio.CancelledError:
                logger.info("Tâche WebSocket annulée")
                break
            except Exception as e:
                logger.warning(f"Erreur inattendue ({e}). Reconnexion dans {retry_delay}s...")

            self._ws = None

            if not self._running:
                logger.info("Arrêt demandé — fin de la boucle de connexion")
                break

            logger.info(f"Reconnexion dans {retry_delay}s...")
            await asyncio.sleep(retry_delay)

    async def disconnect(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("Twelve Data WebSocket déconnecté")


class TwelveDataRestProvider:
    """
    Provider REST pour l'historique via Twelve Data API.
    Utilisé pour charger les bougies passées au démarrage.
    
    Endpoint: https://api.twelvedata.com/time_series
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"

    def fetch_rates(self, symbol: str, count: int) -> pd.DataFrame:
        """
        Récupère l'historique des bougies 1 minute.
        Retourne un DataFrame avec colonnes: time, open, high, low, close, volume
        """
        # TwelveData utilise le format EUR/USD avec slash
        td_symbol = symbol if "/" in symbol else f"{symbol[:3]}/{symbol[3:]}"

        params = {
            "symbol": td_symbol,
            "interval": "1min",
            "outputsize": min(count, 5000),
            "apikey": self.api_key,
        }

        try:
            resp = requests.get(
                f"{self.base_url}/time_series",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "error":
                logger.error(f"Erreur Twelve Data API: {data.get('message', 'inconnue')}")
                return pd.DataFrame()

            values = data.get("values", [])
            if not values:
                logger.warning("Aucune donnée historique reçue de Twelve Data")
                return pd.DataFrame()

            # Conversion en DataFrame
            rows = []
            for v in values:
                rows.append({
                    "time": v["datetime"],
                    "open": float(v["open"]),
                    "high": float(v["high"]),
                    "low": float(v["low"]),
                    "close": float(v["close"]),
                    "volume": float(v.get("volume", 0)),
                })

            df = pd.DataFrame(rows)
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)

            logger.info(
                f"Historique Twelve Data chargé: {len(df)} bougies "
                f"({df['time'].iloc[0]} -> {df['time'].iloc[-1]})"
            )
            return df.tail(count).reset_index(drop=True)

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur réseau Twelve Data: {e}")
            return pd.DataFrame()
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Erreur parsing Twelve Data: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Récupère le prix actuel via l'endpoint /price."""
        td_symbol = symbol if "/" in symbol else f"{symbol[:3]}/{symbol[3:]}"
        try:
            resp = requests.get(
                f"{self.base_url}/price",
                params={"symbol": td_symbol, "apikey": self.api_key},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("price", 0))
        except Exception as e:
            logger.error(f"Erreur prix Twelve Data: {e}")
            return 0.0
