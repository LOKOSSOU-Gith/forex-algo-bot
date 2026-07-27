import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class DiscordNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def _post(self, payload: Dict) -> bool:
        try:
            if HAS_REQUESTS:
                resp = _requests.post(
                    self.webhook_url, json=payload, timeout=10,
                    headers={"User-Agent": "ForexAlgoBot/1.0"},
                )
            else:
                import urllib.request as req
                data = json.dumps(payload).encode()
                req.Request(
                    self.webhook_url, data=data,
                    headers={"Content-Type": "application/json",
                             "User-Agent": "ForexAlgoBot/1.0"},
                    method="POST",
                )
            return True
        except Exception as e:
            logger.error(f"Erreur Discord: {e}")
            return False

    def send_prediction(self, p: Dict) -> bool:
        msg = self._format_prediction(p)
        ok = self._post(msg)
        if ok:
            logger.info("Prediction envoyee a Discord")
        return ok

    def send_verification(self, pred: Dict, result: Dict) -> bool:
        msg = self._format_verification(pred, result)
        ok = self._post(msg)
        if ok:
            logger.info("Verification envoyee a Discord")
        return ok

    def _format_prediction(self, p: Dict) -> Dict:
        up = p["direction"] == "PLUS_HAUT"
        emoji = "🟢" if up else "🔴"
        label = "PLUS HAUT" if up else "PLUS BAS"

        lines = [
            f"{emoji} **EUR/USD — {label} dans 2 min**",
            "",
            f"Prix: {p['current_price']:.5f} | Score: {p['score']:+d}",
            f"EMA20: {p.get('ema',0):.5f} | RSI7: {p.get('rsi',0):.1f}",
            f"ATR: {p.get('atr',0):.5f} | Confiance: {p['confidence']}%",
            "",
            "**Analyse:**",
        ]
        for d in p["details"]:
            sign = f"_{+d['weight']:+d}_" if d["weight"] != 0 else ""
            lines.append(f"{d['icon']} {d['label']}: {d['value']} {sign}".strip())

        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━")
        return {"content": "\n".join(lines), "username": "Forex Algo Bot"}

    def _format_verification(self, pred: Dict, result: Dict) -> Dict:
        up = pred["direction"] == "PLUS_HAUT"
        label = "PLUS HAUT" if up else "PLUS BAS"
        emoji = "✅" if result["correct"] else "❌"
        verdict = "CORRECT" if result["correct"] else "FAUX"
        arrow = "📈" if result["change_pips"] > 0 else "📉"

        lines = [
            f"{emoji} **Verification prediction — {verdict}**",
            "",
            f"Prediction: {label} dans 2 min",
            f"Prix T+0: {result['price_at_prediction']:.5f}",
            f"Prix T+2: {result['current_price']:.5f}",
            f"{arrow} Variation: {result['change_pips']:+.1f} pips",
            "",
            "━━━━━━━━━━━━━━━━━━",
        ]
        return {"content": "\n".join(lines), "username": "Forex Algo Bot"}
