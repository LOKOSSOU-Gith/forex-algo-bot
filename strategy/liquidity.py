import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class LiquidityAnalyzer:
    """
    Détecte les zones de liquidité et les liquidity grabs (chasses aux stops).
    """

    def __init__(self, lookback: int = 48):
        self.lookback = lookback

    def find_swing_highs_lows(self, df: pd.DataFrame) -> Dict[str, float]:
        window = self.lookback
        if len(df) < window:
            window = len(df) // 2
        recent = df.tail(window)
        return {
            "swing_high": float(recent["high"].max()),
            "swing_low": float(recent["low"].min()),
            "current_high": float(df["high"].iloc[-1]),
            "current_low": float(df["low"].iloc[-1]),
            "prev_high": float(df["high"].iloc[-window // 2:-1].max()) if len(df) > window // 2 else float(df["high"].iloc[0]),
            "prev_low": float(df["low"].iloc[-window // 2:-1].min()) if len(df) > window // 2 else float(df["low"].iloc[0]),
        }

    def detect_liquidity_grab(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Détecte un liquidity grab:
        1. Le prix casse un plus haut récent (swing_high)
        2. Puis clôture en dessous de ce niveau dans les N bougies suivantes
        Symétrique pour les plus bas.
        """
        if len(df) < 10:
            return None

        zones = self.find_swing_highs_lows(df)
        swing_high = zones["swing_high"]
        swing_low = zones["swing_low"]

        last_5 = df.tail(5)
        grab_high = None
        grab_low = None

        # Vérifie si le prix a cassé puis rejeté le swing high
        if any(last_5["high"].iloc[:-1] > swing_high):
            close_after = last_5["close"].iloc[-1]
            if close_after < swing_high:
                grab_high = {
                    "type": "sell",
                    "level": swing_high,
                    "message": f"Liquidity grab HAUT détecté (cassure {swing_high:.5f} → retour sous)"
                }

        # Vérifie si le prix a cassé puis rejeté le swing low
        if any(last_5["low"].iloc[:-1] < swing_low):
            close_after = last_5["close"].iloc[-1]
            if close_after > swing_low:
                grab_low = {
                    "type": "buy",
                    "level": swing_low,
                    "message": f"Liquidity grab BAS détecté (cassure {swing_low:.5f} → retour dessus)"
                }

        return grab_high or grab_low

    def get_liquidity_zones(self, df: pd.DataFrame) -> List[Dict]:
        zones = self.find_swing_highs_lows(df)
        return [
            {"type": "resistance", "level": zones["swing_high"], "strength": "strong"},
            {"type": "support", "level": zones["swing_low"], "strength": "strong"},
            {"type": "resistance", "level": zones["prev_high"], "strength": "medium"},
            {"type": "support", "level": zones["prev_low"], "strength": "medium"},
        ]
