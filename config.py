import os
from pathlib import Path

# ─── Trading ────────────────────────────────────────────────────────────────
SYMBOL = "EURUSD"
TIMEFRAME = "1m"
ANALYSIS_INTERVAL = 5
LOOKBACK_CANDLES = 500

# ─── Sessions (Heures GMT) ──────────────────────────────────────────────────
SESSIONS = {
    "Asian":     (0, 9),
    "London":    (8, 17),
    "New_York":  (13, 22),
}

# ─── Seuils de scoring ──────────────────────────────────────────────────────
SCORE_THRESHOLD_OLD = 5   # seuil ancienne strategie (comptage bougies)
MIN_ATR = 0.00005         # ATR minimum pour trader (filtre volatilité)
LIQUIDITY_LOOKBACK = 48
ATR_PERIOD = 14

# ─── Machine Learning ───────────────────────────────────────────────────────
ML_ENABLED = True
ML_MIN_SAMPLES = 30    # echantillons minimum avant 1er entrainement
ML_RETRAIN_EVERY = 10  # retrain tous les N nouveaux echantillons

# ─── Discord ────────────────────────────────────────────────────────────────
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1433132595436322948/Bo4yK5Vplp6ciaXS6Ya7jH_cE36KIGVkKuHoJlSaHsZhEexn5F9RhOWPUmgxI6b3jgIx"

# ─── Data provider REST (yfinance pour historique) ─────────────────────────
DATA_PROVIDER = "yfinance"

# ─── Provider temps réel ────────────────────────────────────────────────────
# Options: "mt5", "oanda", "ctrader", "twelvedata", "yfinance_poll", "dukascopy"
WS_DATA_PROVIDER = "dukascopy"

# ─── MetaTrader 5 (terminal local) ──────────────────────────────────────────
# MT5 doit être installé, lancé et connecté à un compte sur ce PC
MT5_SYMBOL = "EURUSD"
MT5_POLL_MS = 200  # ms entre chaque polling tick

# ─── Twelve Data API (temps réel & historique) ─────────────────────────────
# Crée un compte gratuit: https://twelvedata.com/
# Ta clé personnelle:
TWELVEDATA_API_KEY = "4222eb37333c48af85ef0c0b5b0266aa"
TWELVEDATA_SYMBOL = "EUR/USD"

# ─── OANDA API (compte démo gratuit) ────────────────────────────────────────
# Crée un compte: https://www.oanda.com/demo-account/
# Obtenir le token: https://www.oanda.com/account/statement/api/
OANDA_API_TOKEN = os.getenv("OANDA_API_TOKEN", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_PRACTICE = True  # True = démo, False = réel

# ─── Dukascopy (gratuit, pas d'inscription) ─────────────────────────────────
# Données Forex 1-minute gratuites via dukascopy-python
DUKASCOPY_INSTRUMENT = "EUR/USD"
DUKASCOPY_POLL_INTERVAL = 60  # secondes entre chaque poll

# ─── cTrader WebSocket (optionnel) ──────────────────────────────────────────
# Crée une app: https://connect.spotware.com/apps
CTRADER_CLIENT_ID = os.getenv("CTRADER_CLIENT_ID", "")
CTRADER_CLIENT_SECRET = os.getenv("CTRADER_CLIENT_SECRET", "")
CTRADER_ACCESS_TOKEN = os.getenv("CTRADER_ACCESS_TOKEN", "")
CTRADER_DEMO = True

# ─── Chemins ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "bot.log"
