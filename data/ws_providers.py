import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Callable, Optional, List

logger = logging.getLogger(__name__)


class CandleBuilder:
    """Agrège les ticks en bougies 1-minute."""

    def __init__(self, on_candle_close: Callable, symbol: str = "EURUSD"):
        self.on_candle_close = on_candle_close
        self.symbol = symbol
        self._candles: list[dict] = []
        self._current: Optional[dict] = None
        self._current_minute: Optional[int] = None

    def load_history(self, df) -> None:
        for _, row in df.iterrows():
            self._candles.append({
                "time": row["time"] if hasattr(row, "time") else row.name,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            })
        if self._candles:
            logger.info(f"Historique chargé: {len(self._candles)} bougies")

    def load_candles(self, candles: list) -> None:
        self._candles = candles
        logger.info(f"Historique injecté: {len(self._candles)} bougies")

    def on_tick(self, price: float, ts: Optional[datetime] = None) -> None:
        now = ts or datetime.now(timezone.utc)
        minute = now.minute

        if self._current_minute != minute:
            if self._current is not None:
                self._current["close"] = price
                self._candles.append(self._current)
                if len(self._candles) > 1000:
                    self._candles = self._candles[-500:]
                self._current_minute = minute
                self.on_candle_close(list(self._candles))
            self._current = {
                "time": now.replace(second=0, microsecond=0),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0,
            }
            self._current_minute = minute
        else:
            if self._current is not None:
                self._current["high"] = max(self._current["high"], price)
                self._current["low"] = min(self._current["low"], price)
                self._current["close"] = price

    def add_complete_candle(self, candle: dict) -> None:
        """Ajoute une bougie 1-minute complète (depuis un polling REST).
        Déclenche on_candle_close comme une bougie fermée par ticks."""
        if self._candles and candle["time"] <= self._candles[-1]["time"]:
            return  # Déjà présente
        self._current = None
        self._current_minute = None
        self._candles.append(candle)
        if len(self._candles) > 1000:
            self._candles = self._candles[-500:]
        self.on_candle_close(list(self._candles))

    def get_candles(self, n: int = 100) -> List[dict]:
        candles = (self._candles + [self._current]) if self._current else self._candles
        return candles[-n:]


class WebSocketProvider(ABC):
    @abstractmethod
    async def connect(self, candle_builder: CandleBuilder) -> None:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...


class OANDAProvider(WebSocketProvider):
    """WebSocket OANDA v20 — nécessite un compte démo gratuit."""

    def __init__(self, api_token: str = "", account_id: str = "", practice: bool = True):
        self.api_token = api_token
        self.account_id = account_id
        self.practice = practice
        self._ws = None
        self._running = False

    def _url(self) -> str:
        host = "stream-fxpractice.oanda.com" if self.practice else "stream-fxtrade.oanda.com"
        return f"wss://{host}/v3/accounts/{self.account_id}/pricing/stream?instruments=EUR_USD"

    async def connect(self, candle_builder: CandleBuilder) -> None:
        import websockets

        retry_delay = 5

        while True:
            try:
                headers = {"Authorization": f"Bearer {self.api_token}"}
                url = self._url()
                logger.info(f"Connexion OANDA WebSocket: {url[:80]}...")

                async with websockets.connect(url, extra_headers=headers) as ws:
                    self._ws = ws
                    self._running = True

                    async for raw in ws:
                        if not raw.strip():
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if msg.get("type") == "PRICE":
                            bid = float(msg["closeoutBid"])
                            ask = float(msg["closeoutAsk"])
                            mid = (bid + ask) / 2
                            candle_builder.on_tick(mid)

            except websockets.exceptions.WebSocketException as e:
                logger.warning(f"WebSocket OANDA déconnecté ({e}). Reconnexion dans {retry_delay}s...")
            except asyncio.CancelledError:
                logger.info("Tâche WebSocket OANDA annulée")
                break
            except Exception as e:
                logger.warning(f"Erreur inattendue OANDA ({e}). Reconnexion dans {retry_delay}s...")

            self._ws = None

            if not self._running:
                logger.info("Arrêt demandé — fin de la boucle OANDA")
                break

            logger.info(f"Reconnexion OANDA dans {retry_delay}s...")
            await asyncio.sleep(retry_delay)

    async def disconnect(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("OANDA WebSocket déconnecté")


class CTraderWSProvider(WebSocketProvider):
    """WebSocket cTrader Open API — nécessite une app Spotware Connect."""

    def __init__(self, client_id: str = "", client_secret: str = "", access_token: str = "",
                 demo: bool = True):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.demo = demo
        self._ws = None

    def _url(self) -> str:
        host = "demo.ctraderapi.com" if self.demo else "live.ctraderapi.com"
        return f"wss://{host}/v2"

    async def connect(self, candle_builder: CandleBuilder) -> None:
        import websockets
        url = self._url()
        logger.info(f"Connexion cTrader WebSocket: {url}")
        async with websockets.connect(url) as ws:
            self._ws = ws
            auth = json.dumps({
                "clientId": self.client_id,
                "clientSecret": self.client_secret,
                "accessToken": self.access_token,
            })
            await ws.send(auth)
            resp = await ws.recv()
            logger.info(f"Auth cTrader: {resp[:100]}")

            # Abonnement aux cotations EUR/USD
            sub = json.dumps({
                "type": "spot-subscribe",
                "symbol": "EURUSD",
            })
            await ws.send(sub)

            async for raw in ws:
                if not raw.strip():
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                # Format cTrader: {"type":"quote","symbol":"EURUSD","bid":...,"ask":...}
                if msg.get("type") == "quote" and msg.get("symbol") == "EURUSD":
                    mid = (float(msg["bid"]) + float(msg["ask"])) / 2
                    candle_builder.on_tick(mid)

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None
