import asyncio
import logging
import sys
import warnings

# Filtrer TOUS les DeprecationWarning AVANT tout import
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import config
from data.provider import YFinanceProvider
from data.ws_providers import CandleBuilder, OANDAProvider, CTraderWSProvider
from data.mt5_provider import MT5Provider
from data.twelve_data_provider import TwelveDataWSProvider, TwelveDataRestProvider
from data.yfinance_poll_provider import YFinancePollProvider
from data.dukascopy_provider import DukascopyPollProvider, fetch_history
from strategy.engine import PredictionEngine
from notifier.discord import DiscordNotifier
from learning.memory import LearningDB
from learning.model import MLModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")


class PredictionTracker:
    def __init__(self, notifier: DiscordNotifier, engine: PredictionEngine,
                 memory: LearningDB, model: MLModel):
        self.notifier = notifier
        self.engine = engine
        self.memory = memory
        self.model = model
        self._pending = []
        self._last_dir = None
        self._total_verified = 0

    def add(self, prediction: dict):
        self._pending.append(prediction)
        self._last_dir = prediction["direction"]

    def is_duplicate(self, direction: str) -> bool:
        if self._last_dir == direction:
            return True
        return False

    async def verify_pending(self, get_price):
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        still = []
        for pred in self._pending:
            if now >= pred["verify_at"]:
                price = get_price()
                result = self.engine.verify(pred, price)
                self.memory.record(pred, result["correct"])
                self._total_verified += 1

                # Entrainement ML tous les 10 nouveaux echantillons
                if (self._total_verified % 10 == 0
                        and self._total_verified >= self.model.min_samples):
                    logger.info("Retraining du modele ML...")
                    await asyncio.to_thread(self.model.train, self.memory.history)

                loop = asyncio.get_event_loop()
                ok = await loop.run_in_executor(
                    None, self.notifier.send_verification, pred, result
                )
                if ok:
                    logger.info(
                        f"Verif: {'OK' if result['correct'] else 'KO'} "
                        f"({result['change_pips']:+.1f} pips)"
                    )
            else:
                still.append(pred)
        self._pending = still


def get_ws_provider():
    if config.WS_DATA_PROVIDER == "mt5":
        return MT5Provider(symbol=config.MT5_SYMBOL, poll_ms=config.MT5_POLL_MS)
    elif config.WS_DATA_PROVIDER == "twelvedata":
        return TwelveDataWSProvider(
            api_key=config.TWELVEDATA_API_KEY,
            symbol=config.TWELVEDATA_SYMBOL,
        )
    elif config.WS_DATA_PROVIDER == "oanda":
        return OANDAProvider(
            api_token=config.OANDA_API_TOKEN,
            account_id=config.OANDA_ACCOUNT_ID,
            practice=config.OANDA_PRACTICE,
        )
    elif config.WS_DATA_PROVIDER == "ctrader":
        return CTraderWSProvider(
            client_id=config.CTRADER_CLIENT_ID,
            client_secret=config.CTRADER_CLIENT_SECRET,
            access_token=config.CTRADER_ACCESS_TOKEN,
            demo=config.CTRADER_DEMO,
        )
    elif config.WS_DATA_PROVIDER == "yfinance_poll":
        return YFinancePollProvider(
            symbol=config.SYMBOL,
            poll_interval=60,
        )
    elif config.WS_DATA_PROVIDER == "dukascopy":
        return DukascopyPollProvider(
            instrument=config.DUKASCOPY_INSTRUMENT,
            poll_interval=config.DUKASCOPY_POLL_INTERVAL,
        )
    raise ValueError(f"Provider inconnu: {config.WS_DATA_PROVIDER}")


async def on_candle_close(
    candles: list,
    engine: PredictionEngine,
    notifier: DiscordNotifier,
    tracker: PredictionTracker,
    memory: LearningDB,
    model: MLModel,
):
    if len(candles) < 20:
        return

    import pandas as pd
    df = pd.DataFrame(candles[-config.LOOKBACK_CANDLES:])

    prediction = engine.predict_2min(df)
    if prediction is None:
        return

    # ML override: si le modele est confiant, il remplace la direction
    override = model.should_override(prediction)
    if override:
        old_dir = prediction["direction"]
        prediction["direction"] = override
        logger.info(
            f"ML override: {old_dir} -> {override} "
            f"(pattern={prediction.get('green_red', '?')})"
        )

    logger.info(
        f"Pred: {prediction['direction']} | "
        f"Score: {prediction['score']} | "
        f"Pattern: {prediction.get('green_red', '?')} | "
        f"Session: {prediction['session']}"
    )

    if tracker.is_duplicate(prediction["direction"]):
        logger.info("Duplicate ignore")
        return

    tracker.add(prediction)

    loop = asyncio.get_event_loop()
    sent = await loop.run_in_executor(None, notifier.send_prediction, prediction)
    if sent:
        logger.info("Prediction envoyee")


async def verify_loop(tracker: PredictionTracker, candle_builder: CandleBuilder):
    while True:
        await asyncio.sleep(30)
        if not tracker._pending:
            continue
        try:
            candles = candle_builder.get_candles(3)
            if candles:
                price = candles[-1]["close"]
                await tracker.verify_pending(lambda: price)
        except Exception as e:
            logger.warning(f"Erreur verif: {e}")


async def memory_flush_loop(memory: LearningDB):
    """Sauvegarde forcee de la memoire toutes les 30 secondes."""
    while True:
        await asyncio.sleep(30)
        memory.flush()


async def main():
    logger.info("=" * 50)
    logger.info("Forex Algo Bot — 2 Strategies Combinees")
    logger.info(f"Provider: {config.WS_DATA_PROVIDER}")
    logger.info(f"Bougies: seuil {config.SCORE_THRESHOLD_OLD}")
    logger.info(f"Confirmation: EMA20 + RSI7 | Min ATR: {config.MIN_ATR}")
    logger.info(f"ML: {config.ML_ENABLED}")
    logger.info("=" * 50)

    memory = LearningDB(str(config.BASE_DIR / "memory.json"))
    logger.info(f"Memoire: {memory.count()} antecedents")

    model = MLModel(
        model_path=str(config.BASE_DIR / "model.pkl"),
        min_samples=config.ML_MIN_SAMPLES,
    )
    if config.ML_ENABLED:
        loaded = model._load()
        if not loaded and memory.count() >= model.min_samples:
            logger.info("Entrainement initial du modele ML...")
            await asyncio.to_thread(model.train, memory.history)

    engine = PredictionEngine(
        score_threshold=config.SCORE_THRESHOLD_OLD,
        min_atr=config.MIN_ATR,
    )
    notifier = DiscordNotifier(config.DISCORD_WEBHOOK_URL)
    tracker = PredictionTracker(notifier, engine, memory, model)

    cb = CandleBuilder(
        on_candle_close=lambda candles: asyncio.create_task(
            on_candle_close(candles, engine, notifier, tracker, memory, model)
        ),
        symbol=config.SYMBOL,
    )

    # Chargement historique
    if config.WS_DATA_PROVIDER != "mt5":
        if config.WS_DATA_PROVIDER == "twelvedata":
            logger.info("Chargement historique (Twelve Data)...")
            h = TwelveDataRestProvider(api_key=config.TWELVEDATA_API_KEY)
            df = h.fetch_rates(config.SYMBOL, config.LOOKBACK_CANDLES)
        elif config.WS_DATA_PROVIDER == "dukascopy":
            logger.info("Chargement historique (Dukascopy)...")
            df = await fetch_history(
                instrument=config.DUKASCOPY_INSTRUMENT,
                count=config.LOOKBACK_CANDLES,
            )
        else:
            logger.info("Chargement historique (yfinance)...")
            h = YFinanceProvider()
            df = h.fetch_rates(config.SYMBOL, config.LOOKBACK_CANDLES)

        if df.empty:
            logger.error("Echec historique")
            memory.close()
            return
        cb.load_history(df)

    ws = get_ws_provider()
    logger.info(f"Connexion ({config.WS_DATA_PROVIDER})...")

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(ws.connect(cb))
            tg.create_task(verify_loop(tracker, cb))
            tg.create_task(memory_flush_loop(memory))
    finally:
        memory.close()


if __name__ == "__main__":
    asyncio.run(main())
