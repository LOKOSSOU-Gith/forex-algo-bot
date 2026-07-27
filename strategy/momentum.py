import pandas as pd
import numpy as np
from typing import Dict, Tuple


class MomentumAnalyzer:
    """
    Analyse technique M1 : RSI, EMA, ATR, momentum.
    """

    def __init__(self, atr_period: int = 14, rsi_period: int = 7, ema_period: int = 20):
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.ema_period = ema_period

    def atr(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return 0.0
        high, low, close = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        return float(np.mean(tr[-self.atr_period:])) if len(tr) >= self.atr_period else float(np.mean(tr))

    def ema(self, df: pd.DataFrame) -> float:
        """Retourne la valeur EMA20 actuelle."""
        closes = df["close"].values
        if len(closes) < self.ema_period:
            return float(closes[-1])
        series = pd.Series(closes)
        return float(series.ewm(span=self.ema_period, adjust=False).mean().iloc[-1])

    def rsi(self, df: pd.DataFrame) -> float:
        """Retourne RSI 7 (Wilder)."""
        closes = df["close"].values
        if len(closes) < self.rsi_period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[:self.rsi_period])
        avg_loss = np.mean(losses[:self.rsi_period])
        for i in range(self.rsi_period, len(gains)):
            avg_gain = (avg_gain * (self.rsi_period - 1) + gains[i]) / self.rsi_period
            avg_loss = (avg_loss * (self.rsi_period - 1) + losses[i]) / self.rsi_period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def candle_bias(self, candle) -> int:
        """Retourne +1 pour bougie haussiere, -1 pour baissiere, 0 pour doji."""
        body = candle["close"] - candle["open"]
        if abs(body) < 0.0001:
            return 0
        return 1 if body > 0 else -1

    def consecutive_momentum(self, df: pd.DataFrame, min_consecutive: int = 3) -> Dict:
        """Analyse les bougies consecutives dans la meme direction."""
        if len(df) < min_consecutive:
            return {"direction": None, "count": 0, "strength": 0}
        recent = df.tail(min_consecutive + 5)
        biases = [self.candle_bias(recent.iloc[i]) for i in range(len(recent))]
        count = 0
        direction = None
        for b in reversed(biases):
            if b == 0:
                break
            if direction is None:
                direction = b
            if b == direction:
                count += 1
            else:
                break
        strength = min(count / min_consecutive, 1.0) if count > 0 else 0
        return {
            "direction": "buy" if direction == 1 else ("sell" if direction == -1 else None),
            "count": count,
            "strength": round(strength, 2),
        }

    def acceleration(self, df: pd.DataFrame) -> Dict:
        """Detecte si les bougies s'accelerent."""
        if len(df) < 5:
            return {"accelerating": False, "factor": 0.0}
        last_5_bodies = []
        for i in range(-5, 0):
            body = abs(df["close"].iloc[i] - df["open"].iloc[i])
            last_5_bodies.append(body)
        avg_early = np.mean(last_5_bodies[:3])
        avg_late = np.mean(last_5_bodies[-2:])
        factor = avg_late / avg_early if avg_early > 0 else 0
        return {"accelerating": factor > 1.3, "factor": round(factor, 2)}

    def analyze_base(self, df: pd.DataFrame) -> Dict:
        """Analyse momentum de l'ancienne strategie (consecutives + acceleration)."""
        momentum = self.consecutive_momentum(df)
        accel = self.acceleration(df)
        atr_val = self.atr(df)
        score = 0
        if momentum["direction"] == "buy":
            score += momentum["count"]
            if accel["accelerating"]:
                score += 2
        elif momentum["direction"] == "sell":
            score -= momentum["count"]
            if accel["accelerating"]:
                score -= 2
        return {"momentum": momentum, "acceleration": accel, "atr": round(atr_val, 5), "score": score}

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyse complete : RSI, EMA, ATR."""
        return {
            "atr": round(self.atr(df), 5),
            "ema": round(self.ema(df), 5),
            "rsi": round(self.rsi(df), 1),
        }
