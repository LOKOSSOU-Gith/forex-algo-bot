from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def fetch_rates(self, symbol: str, count: int) -> pd.DataFrame:
        """Retourne un DataFrame avec colonnes: time, open, high, low, close, volume"""

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        ...


class YFinanceProvider(DataProvider):
    def __init__(self):
        import yfinance as yf
        self.yf = yf

    def _ticker(self, symbol: str) -> str:
        return f"{symbol}=X"

    def fetch_rates(self, symbol: str, count: int) -> pd.DataFrame:
        ticker = self.yf.Ticker(self._ticker(symbol))
        df = ticker.history(period="5d", interval="1m")
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "Datetime": "time", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        df.index.name = "time"
        df = df[["open", "high", "low", "close", "volume"]]
        df = df.reset_index()
        df["time"] = df["time"].dt.tz_localize(None)
        return df.tail(count).reset_index(drop=True)

    def get_current_price(self, symbol: str) -> float:
        ticker = self.yf.Ticker(self._ticker(symbol))
        data = ticker.history(period="1d", interval="1m")
        return float(data["Close"].iloc[-1])


class CTraderProvider(DataProvider):
    """
    Provider cTrader via Open API.
    Configuration requise dans config.py :
      CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN
    """

    def __init__(self, client_id: str = "", client_secret: str = "", token: str = ""):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = token
        self._ws = None

    def _ensure_auth(self):
        if not self.token:
            raise RuntimeError(
                "cTrader non configuré. Configure CTRADER_CLIENT_ID, "
                "CTRADER_CLIENT_SECRET et CTRADER_ACCESS_TOKEN dans config.py"
            )

    def fetch_rates(self, symbol: str, count: int) -> pd.DataFrame:
        self._ensure_auth()
        # TODO: implémenter l'appel REST cTrader Open API
        # https://connect.spotware.com/docs/open_api_rest
        raise NotImplementedError(
            "CTraderProvider.fetch_rates non implémenté. "
            "Utilise YFinanceProvider (config.DATA_PROVIDER = 'yfinance') "
            "en attendant l'intégration complète."
        )

    def get_current_price(self, symbol: str) -> float:
        self._ensure_auth()
        raise NotImplementedError("CTraderProvider.get_current_price non implémenté")
