import json
import logging
import atexit
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class LearningDB:
    MAX_HISTORY = 500  # nombre max d'entrees conservees

    def __init__(self, path: str):
        self.path = Path(path)
        self.history: list = []
        self._dirty = False
        self._closed = False
        self._load()
        # Sauvegarde automatique a la sortie du processus, meme si close() non appelee
        atexit.register(self._atexit_save)

    @staticmethod
    def profile(prediction: dict) -> str:
        gr = prediction.get("green_red", "?g?r")
        sess = prediction.get("session", "?")
        ov = "X" if prediction.get("overlap") else ""
        return f"{gr}|{sess}{ov}"

    @staticmethod
    def features(prediction: dict) -> dict:
        gr = prediction.get("green_red", "0g3r")
        try:
            g = int(gr.split("g")[0])
            r = int(gr.split("g")[1].split("r")[0])
        except (ValueError, IndexError):
            g, r = 0, 3
        current_price = prediction.get("current_price", 0)
        ema_val = prediction.get("ema", 0)
        return {
            "green_count": g,
            "red_count": r,
            "direction": prediction.get("direction", "?"),
            "score": prediction.get("score", 0),
            "session": prediction.get("session", "?"),
            "overlap": bool(prediction.get("overlap")),
            "atr": prediction.get("atr", 0),
            "rsi": prediction.get("rsi", 50),
            "price_ema_diff": current_price - ema_val,
            "hour": prediction.get("timestamp").hour
                if hasattr(prediction.get("timestamp"), "hour")
                else 0,
        }

    def _atexit_save(self):
        """Sauvegarde de secours via atexit (si close() n'a pas ete appelee)."""
        if not self._closed and self._dirty:
            self._save()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Toujours un historique de type liste
                if isinstance(data, list):
                    self.history = data[-self.MAX_HISTORY:]
                else:
                    self.history = []
                logger.info(f"Memoire chargee: {len(self.history)} entrees")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Erreur chargement memoire: {e}")
                self.history = []

    def _save(self):
        """Sauvegarde atomique : fichier temp -> renommage.
        Evite la corruption si le processus est interrompu pendant l'ecriture."""
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Ecriture dans un fichier temporaire puis renommage atomique
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.path.parent),
                prefix="memory_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self.history[-self.MAX_HISTORY:], f, indent=2, default=str)
                os.replace(tmp_path, str(self.path))
            except Exception:
                # Nettoyer le fichier temp en cas d'erreur
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            self._dirty = False
        except (IOError, OSError) as e:
            logger.error(f"Erreur sauvegarde memoire: {e}")

    def record(self, prediction: dict, outcome: bool) -> None:
        # Direction reelle du prix dans 2 min
        if prediction["direction"] == "PLUS_HAUT":
            actual = "UP" if outcome else "DOWN"
        else:
            actual = "DOWN" if outcome else "UP"

        entry = {
            "profile": self.profile(prediction),
            "features": self.features(prediction),
            "outcome": outcome,
            "actual_direction": actual,
        }
        self.history.append(entry)
        self._dirty = True
        # Sauvegarde APRES CHAQUE enregistrement, pas seulement toutes les 10
        self._save()
        prof = entry["profile"]
        wr = self.winrate(prof)
        n = self.count(prof)
        logger.info(
            f"Apprentissage: {prof} -> {'OK' if outcome else 'KO'} "
            f"(n={n}, winrate={wr:.0%}, reel={actual})"
        )

    def count(self, profile: str = None) -> int:
        if profile is None:
            return len(self.history)
        return sum(1 for e in self.history if e["profile"] == profile)

    def winrate(self, profile: str = None) -> float:
        entries = self.history if profile is None else \
            [e for e in self.history if e["profile"] == profile]
        if not entries:
            return 0.0
        return sum(1 for e in entries if e["outcome"]) / len(entries)

    def should_skip(self, prediction: dict) -> bool:
        prof = self.profile(prediction)
        n = self.count(prof)
        wr = self.winrate(prof)
        if n < 5:
            return False
        if wr < 0.40:
            logger.info(f"Evite {prof}: winrate {wr:.0%} sur {n} essais")
            return True
        return False

    def flush(self):
        """Sauvegarde forcee (appele periodiquement par la boucle de verification)."""
        if self._dirty:
            self._save()

    def summary(self) -> str:
        profiles = set(e["profile"] for e in self.history)
        lines = []
        for p in sorted(profiles):
            n = self.count(p)
            wr = self.winrate(p)
            lines.append(f"{p}: {n} essais, {wr:.0%} reussite")
        return "\n".join(lines)

    def close(self):
        self._closed = True
        self._save()
        # Nettoyer le handler atexit pour eviter un double appel
        atexit.unregister(self._atexit_save)
