import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MLModel:
    """
    Modele RandomForest qui apprend a predire la direction du prix
    dans 2 minutes a partir des caracteristiques des bougies.
    """

    FEATURE_NAMES = [
        "green_count", "red_count", "score", "atr", "rsi",
        "price_ema_diff", "hour",
        "session_Asian", "session_London", "session_NewYork",
        "overlap",
    ]

    OVERRIDE_CONFIDENCE = 0.60  # confiance minimale pour remplacer la regle

    def __init__(self, model_path: str, min_samples: int = 50):
        self.model_path = Path(model_path)
        self.min_samples = min_samples
        self.model: Optional[RandomForestClassifier] = None
        self._last_train_count = 0

    def _vectorize(self, features: dict) -> np.ndarray:
        """Transforme un dict de features en vecteur numpy ordonne."""
        sess = features.get("session", "")
        vals = [
            features.get("green_count", 0),
            features.get("red_count", 0),
            features.get("score", 0),
            features.get("atr", 0),
            features.get("rsi", 50),
            features.get("price_ema_diff", 0),
            features.get("hour", 0),
            1 if sess == "Asian" else 0,
            1 if sess == "London" else 0,
            1 if sess == "New_York" else 0,
            1 if features.get("overlap") else 0,
        ]
        return np.array(vals, dtype=np.float64).reshape(1, -1)

    def _build_dataset(self, history: list):
        X, y = [], []
        for entry in history:
            feat = entry.get("features")
            if not feat:
                continue
            actual = entry.get("actual_direction")
            if actual is None:
                dir_pred = feat.get("direction", "?")
                outcome = entry.get("outcome")
                if outcome is None:
                    continue
                if dir_pred == "PLUS_HAUT":
                    actual = "UP" if outcome else "DOWN"
                else:
                    actual = "DOWN" if outcome else "UP"
            X.append(self._vectorize(feat).flatten())
            y.append(1 if actual == "UP" else 0)
        if not X:
            return None, None
        return np.vstack(X), np.array(y)

    def train(self, history: list) -> bool:
        if not HAS_SKLEARN:
            logger.warning("scikit-learn non installe, modele ML desactive")
            return False

        X, y = self._build_dataset(history)
        if X is None or len(y) < self.min_samples:
            logger.info(
                f"Entrainement ML reporte: {len(y) if y is not None else 0} "
                f"echantillons (< {self.min_samples})"
            )
            return False

        n_classes = len(set(y))
        if n_classes < 2:
            logger.warning(f"Une seule classe dans les donnees ({set(y)}), pas d'entrainement")
            return False

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        )
        self.model.fit(X, y)
        self._last_train_count = len(y)

        importances = sorted(zip(self.FEATURE_NAMES, self.model.feature_importances_),
                             key=lambda x: -x[1])
        top = ", ".join(f"{n}={v:.3f}" for n, v in importances[:5])
        acc = self.model.score(X, y)
        logger.info(
            f"Modele ML entraine: {len(y)} echantillons, "
            f"accuracy={acc:.0%}, top features: {top}"
        )

        self._save()
        return True

    def predict_direction(self, prediction: dict) -> Optional[Tuple[str, float]]:
        """
        Predire la direction du prix dans 2 min a partir des bougies.
        Retourne (direction, probabilite) ou None si modele non entraine.
        direction: 'PLUS_HAUT' ou 'PLUS_BAS'
        probabilite: 0.5 = aleatoire, 1.0 = certain
        """
        if self.model is None:
            return None

        current_price = prediction.get("current_price", 0)
        ema_val = prediction.get("ema", 0)
        price_ema_diff = current_price - ema_val

        feat = {
            "green_count": self._parse_green(prediction.get("green_red", "0g3r")),
            "red_count": self._parse_red(prediction.get("green_red", "0g3r")),
            "score": prediction.get("score", 0),
            "session": prediction.get("session", "?"),
            "overlap": bool(prediction.get("overlap")),
            "atr": prediction.get("atr", 0),
            "rsi": prediction.get("rsi", 50),
            "price_ema_diff": price_ema_diff,
            "hour": prediction.get("timestamp").hour
                if hasattr(prediction.get("timestamp"), "hour") else 0,
        }

        X = self._vectorize(feat)
        probas = self.model.predict_proba(X)[0]
        # probas[0] = probabilite DOWN, probas[1] = probabilite UP
        proba_up = float(probas[1])
        proba_down = float(probas[0])

        if proba_up >= proba_down:
            return ("PLUS_HAUT", proba_up)
        else:
            return ("PLUS_BAS", proba_down)

    def should_override(self, prediction: dict) -> Optional[str]:
        """
        Retourne la direction override si le ML est suffisamment confiant,
        None sinon.
        """
        result = self.predict_direction(prediction)
        if result is None:
            return None

        ml_dir, proba = result
        if proba >= self.OVERRIDE_CONFIDENCE:
            logger.info(
                f"ML override: {prediction.get('green_red','?')} "
                f"→ {ml_dir} (proba={proba:.0%})"
            )
            return ml_dir
        return None

    def _save(self):
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info(f"Modele sauvegarde: {self.model_path}")
        except IOError as e:
            logger.error(f"Erreur sauvegarde modele: {e}")

    def _load(self):
        if self.model_path.exists() and HAS_SKLEARN:
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"Modele charge: {self.model_path}")
                return True
            except (pickle.PickleError, IOError) as e:
                logger.warning(f"Impossible de charger le modele: {e}")
        return False

    @staticmethod
    def _parse_green(green_red: str) -> int:
        try:
            return int(green_red.split("g")[0])
        except (ValueError, IndexError):
            return 0

    @staticmethod
    def _parse_red(green_red: str) -> int:
        try:
            return int(green_red.split("g")[1].split("r")[0])
        except (ValueError, IndexError):
            return 3
