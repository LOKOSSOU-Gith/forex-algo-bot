import pandas as pd
from typing import Optional, Dict
from datetime import datetime, timedelta

from strategy.liquidity import LiquidityAnalyzer
from strategy.momentum import MomentumAnalyzer
from utils.helpers import get_session, get_session_overlap, gmt_now, format_price


class PredictionEngine:
    """
    Moteur de prediction 2 minutes.
    Deux strategies combinees :
      1. Ancienne : comptage bougies + momentum + session (signal de base)
      2. Nouvelle  : EMA20 + RSI7 (confirmation tendance)
    Les deux doivent etre d'accord pour emettre un signal.
    """

    def __init__(self, score_threshold: int = 5, min_atr: float = 0.00005):
        self.liquidity = LiquidityAnalyzer()
        self.momentum = MomentumAnalyzer()
        self.score_threshold = score_threshold
        self.min_atr = min_atr

    # ── Ancienne stratégie : comptage bougies ─────────────────────────────
    def _predict_base(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Signal base sur les 3 dernieres bougies + momentum + session.
        Retourne une prediction partielle ou None.
        """
        if df.empty or len(df) < 5:
            return None

        now = gmt_now()
        current_price = float(df["close"].iloc[-1])
        last_3 = df.tail(3)

        green = sum(1 for _, r in last_3.iterrows() if r["close"] > r["open"])
        red = 3 - green

        mom = self.momentum.analyze_base(df)
        grab = self.liquidity.detect_liquidity_grab(df)
        session = get_session(now.hour)
        overlap = get_session_overlap(now.hour)

        score = 0
        details = []
        direction = None

        if green == 3:
            direction = "PLUS_HAUT"
            score = mom["score"] if mom["score"] > 0 else 2
            if mom["score"] > 0:
                score += mom["score"]
        elif red == 3:
            direction = "PLUS_BAS"
            score = abs(mom["score"]) if mom["score"] < 0 else 2
            if mom["score"] < 0:
                score += abs(mom["score"])
        elif green >= 2:
            direction = "PLUS_HAUT"
            score = 1
        elif red >= 2:
            direction = "PLUS_BAS"
            score = 1
        else:
            return None

        if grab:
            bonus = 2
            score += bonus
            details.append({
                "icon": "🎯", "label": "Liquidité",
                "value": grab["message"][:40], "weight": bonus,
            })

        if overlap:
            score += 1
            details.append({
                "icon": "🕐", "label": "Session",
                "value": overlap, "weight": 1,
            })

        if score < self.score_threshold:
            return None

        green_red = f"{green}g{red}r"

        details.insert(0, {
            "icon": "📊", "label": "Bougies",
            "value": f"{green} vertes / {red} rouges", "weight": score,
        })

        return {
            "direction": direction,
            "score": score,
            "details": details,
            "green_red": green_red,
            "session": session,
            "overlap": overlap,
            "atr": mom["atr"],
        }

    # ── Nouvelle stratégie : confirmation EMA20 + RSI7 ────────────────────
    def _confirm_trend(self, df: pd.DataFrame, base_dir: str) -> Optional[str]:
        """
        Confirme la direction avec EMA20 + RSI7 (flow diagram).
        PLUS_HAUT  → prix > EMA20 ET RSI > 50
        PLUS_BAS   → prix < EMA20 ET RSI < 50
        Retourne la direction ou None si desaccord.
        """
        if df.empty or len(df) < 21:
            return None

        current_price = float(df["close"].iloc[-1])
        ema_val = self.momentum.ema(df)
        atr_val = self.momentum.atr(df)
        rsi_val = self.momentum.rsi(df)

        if atr_val < self.min_atr:
            return None

        if base_dir == "PLUS_HAUT":
            return base_dir if current_price > ema_val and rsi_val > 50 else None

        if base_dir == "PLUS_BAS":
            return base_dir if current_price < ema_val and rsi_val < 50 else None

        return None

    # ── Prediction combinee ──────────────────────────────────────────────
    def predict_2min(self, df: pd.DataFrame) -> Optional[Dict]:
        """Les deux strategies doivent etre d'accord."""
        base = self._predict_base(df)
        if base is None:
            return None

        confirm = self._confirm_trend(df, base["direction"])
        if confirm is None:
            return None

        now = gmt_now()
        current_price = float(df["close"].iloc[-1])
        session = get_session(now.hour)
        overlap = get_session_overlap(now.hour)

        tech = self.momentum.analyze(df)
        zones = self.liquidity.find_swing_highs_lows(df)

        confidence = min(50 + base["score"] * 6, 95)

        details = base["details"]
        details.append({
            "icon": "✅", "label": "Confirmation",
            "value": f"EMA20 + RSI OK", "weight": 0,
        })

        return {
            "timestamp": now,
            "current_price": current_price,
            "direction": base["direction"],
            "confidence": confidence,
            "score": base["score"],
            "details": details,
            "session": session,
            "overlap": overlap,
            "atr": base["atr"],
            "rsi": tech["rsi"],
            "ema": tech["ema"],
            "zones": zones,
            "prediction_price": current_price,
            "prediction_time": now,
            "verify_at": now + timedelta(minutes=2),
            "green_red": base["green_red"],
        }

    def verify(self, prediction: Dict, current_price: float) -> Optional[Dict]:
        if prediction is None:
            return None
        predicted_direction = prediction["direction"]
        price_at_prediction = prediction["prediction_price"]
        correct = False

        if predicted_direction == "PLUS_HAUT" and current_price > price_at_prediction:
            correct = True
        elif predicted_direction == "PLUS_BAS" and current_price < price_at_prediction:
            correct = True

        change_pips = (current_price - price_at_prediction) * 10000

        return {
            "correct": correct,
            "change_pips": round(change_pips, 1),
            "price_at_prediction": price_at_prediction,
            "current_price": current_price,
        }
