#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║     FOREX ALGO BOT — Hugging Face Space                           ║
║     Backtest, visualisation & analyse de performance               ║
║     Stratégie: comptage bougies + EMA20 + RSI7 sur EUR/USD M1     ║
╚══════════════════════════════════════════════════════════════════════╝

Déployé sur Hugging Face Spaces.
Utilise yfinance (gratuit) pour les données historiques et
scikit-learn pour les modèles ML.
"""

import os
import sys
import json
import io
import base64
import math
import random
import warnings
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

# ─── Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# Matplotlib pour les graphiques
try:
    import matplotlib
    matplotlib.use("Agg")  # Mode non-interactif pour HF Spaces
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    plt = None
    Figure = None

import gradio as gr  # HF Spaces installe gradio automatiquement

# Ajouter le répertoire courant au path
_BOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(_BOT_DIR))

# ─── Imports du bot ─────────────────────────────────────────────────────────
try:
    from strategy.engine import PredictionEngine
    from strategy.momentum import MomentumAnalyzer
    from strategy.liquidity import LiquidityAnalyzer
    from learning.model import MLModel, HAS_SKLEARN
    BOT_OK = True
except Exception as e:
    BOT_OK = False
    print(f"[!] Erreur import bot: {e}")

# ─── Paramètres ─────────────────────────────────────────────────────────────
SYMBOL = "EURUSD"
LOOKBACK = 500

# ============================================================
# Data Loading
# ============================================================

def fetch_historical_data(symbol: str = SYMBOL, count: int = 500) -> pd.DataFrame:
    """Fetch historical EUR/USD 1-minute data via yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}=X")
        df = ticker.history(period="5d", interval="1m")
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })
        df.index.name = "time"
        df = df[["open", "high", "low", "close", "volume"]].copy()
        # Gérer le nom de la colonne d'index (yfinance: "Datetime" ou "Date")
        df = df.reset_index()
        time_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else None)
        if time_col and hasattr(df[time_col], "dt"):
            df["time"] = df[time_col]
        elif time_col:
            df["time"] = df[time_col]
        df = df.drop(columns=["Datetime", "Date"], errors="ignore")
        if hasattr(df["time"].dtype, "tz") and df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_localize(None)
        return df.tail(count).reset_index(drop=True)
    except Exception as e:
        print(f"[!] Erreur yfinance: {e}")
        return pd.DataFrame()


def run_backtest(df: pd.DataFrame) -> Tuple[List[Dict], Dict]:
    """Run the prediction engine on historical data and return results."""
    if df.empty or len(df) < 50:
        return [], {}

    engine = PredictionEngine(score_threshold=3, min_atr=0.00001)
    predictions = []
    stats = {
        "total": 0, "correct": 0, "wrong": 0,
        "plus_haut": 0, "plus_bas": 0,
        "haut_correct": 0, "bas_correct": 0,
        "scores": [], "confidences": [],
        "sessions": defaultdict(lambda: {"ok": 0, "total": 0}),
        "hours": defaultdict(lambda: {"ok": 0, "total": 0}),
    }

    for i in range(30, len(df)):
        chunk = df.iloc[:i+1]
        prediction = engine.predict_2min(chunk)
        if prediction is None:
            continue

        # Vérifier si on peut vérifier T+2
        if i + 2 >= len(df):
            continue

        future_price = float(df.iloc[i + 2]["close"])
        result = engine.verify(prediction, future_price)
        if result is None:
            continue

        pred = {
            "time": str(df.iloc[i]["time"]) if hasattr(df.iloc[i], "time") else str(i),
            "price": prediction["current_price"],
            "direction": prediction["direction"],
            "score": prediction["score"],
            "confidence": prediction["confidence"],
            "rsi": prediction.get("rsi", 0),
            "ema": prediction.get("ema", 0),
            "atr": prediction.get("atr", 0),
            "session": prediction.get("session", "?"),
            "green_red": prediction.get("green_red", "?g?r"),
            "correct": result["correct"],
            "change_pips": result["change_pips"],
            "future_price": future_price,
        }
        predictions.append(pred)

        stats["total"] += 1
        stats["scores"].append(prediction["score"])
        stats["confidences"].append(prediction["confidence"])
        if result["correct"]:
            stats["correct"] += 1
        else:
            stats["wrong"] += 1

        if prediction["direction"] == "PLUS_HAUT":
            stats["plus_haut"] += 1
            if result["correct"]:
                stats["haut_correct"] += 1
        else:
            stats["plus_bas"] += 1
            if result["correct"]:
                stats["bas_correct"] += 1

        sess = prediction.get("session", "?")
        stats["sessions"][sess]["total"] += 1
        if result["correct"]:
            stats["sessions"][sess]["ok"] += 1

        hour = prediction.get("timestamp", datetime.now()).hour if hasattr(prediction.get("timestamp"), "hour") else 0
        stats["hours"][hour]["total"] += 1
        if result["correct"]:
            stats["hours"][hour]["ok"] += 1

    return predictions, dict(stats)


def load_memory_data() -> Tuple[List[Dict], Dict]:
    """Load and analyze memory.json if it exists."""
    memory_path = _BOT_DIR / "memory.json"
    if not memory_path.exists():
        return [], {}

    try:
        with open(memory_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        return [], {}

    if not history:
        return [], {}

    total = len(history)
    correct = sum(1 for e in history if e["outcome"])
    stats = {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "winrate": round(correct / total * 100, 1) if total else 0,
        "profiles": defaultdict(lambda: {"ok": 0, "total": 0}),
        "sessions": defaultdict(lambda: {"ok": 0, "total": 0}),
        "directions": defaultdict(lambda: {"ok": 0, "total": 0}),
        "scores_ok": [], "scores_ko": [],
    }

    for e in history:
        prof = e.get("profile", "?")
        feat = e.get("features", {})
        stats["profiles"][prof]["total"] += 1
        if e["outcome"]:
            stats["profiles"][prof]["ok"] += 1

        sess = feat.get("session", "?")
        stats["sessions"][sess]["total"] += 1
        if e["outcome"]:
            stats["sessions"][sess]["ok"] += 1

        d = feat.get("direction", "?")
        stats["directions"][d]["total"] += 1
        if e["outcome"]:
            stats["directions"][d]["ok"] += 1

        score = feat.get("score", 0)
        if e["outcome"]:
            stats["scores_ok"].append(score)
        else:
            stats["scores_ko"].append(score)

    return history, dict(stats)


def train_ml_on_memory(history: List[Dict]) -> Optional[Dict]:
    """Train the ML model on memory data."""
    if not HAS_SKLEARN or not history:
        return None

    model = MLModel(
        model_path=str(_BOT_DIR / "model.pkl"),
        min_samples=10,
    )

    trained = model.train(history)
    if not trained:
        return {"status": "not_enough_data", "samples": len(history)}

    # Test on some predictions
    test_results = []
    for entry in history[-20:]:
        feat = entry.get("features", {})
        if not feat:
            continue
        pred = {
            "direction": feat.get("direction", "PLUS_HAUT"),
            "green_red": f"{feat.get('green_count', 0)}g{feat.get('red_count', 3)}r",
            "score": feat.get("score", 0),
            "session": feat.get("session", "?"),
            "overlap": feat.get("overlap", False),
            "atr": feat.get("atr", 0),
            "rsi": feat.get("rsi", 50),
            "current_price": 1.10,
            "ema": 1.10,
            "timestamp": datetime.now(),
        }
        override = model.should_override(pred)
        test_results.append({
            "actual_outcome": entry["outcome"],
            "ml_override": override is not None,
            "ml_direction": override or "none",
        })

    return {
        "status": "trained",
        "samples": len(history),
        "test_results": test_results[:10],
    }


# ============================================================
# Candlestick Chart with matplotlib
# ============================================================

def generate_candlestick_chart(df: pd.DataFrame,
                                predictions: Optional[List[Dict]] = None,
                                max_candles: int = 80) -> Optional[Figure]:
    """
    Generate a professional candlestick chart with:
    - Green/red candlesticks
    - EMA20 line
    - Prediction markers (arrows) with correct/incorrect coloring
    - Session backgrounds
    - Proper forex price formatting
    """
    if not MATPLOTLIB_OK or df.empty:
        return None

    fig = None
    try:
        # Use last N candles for readability
        chart_df = df.tail(max_candles).copy()
        if len(chart_df) < 10:
            return None

        # Ensure time is datetime
        if not np.issubdtype(chart_df["time"].dtype, np.datetime64):
            chart_df["time"] = pd.to_datetime(chart_df["time"])

        # Calculate EMA20
        closes = chart_df["close"].values
        chart_df["ema20"] = pd.Series(closes).ewm(span=20, adjust=False).mean()

        # Create figure with dark theme
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        # Determine price range for y-axis
        y_min = chart_df["low"].min()
        y_max = chart_df["high"].max()
        y_range = y_max - y_min
        padding = y_range * 0.08
        ax.set_ylim(y_min - padding, y_max + padding)

        # Convert times to matplotlib date numbers
        times = mdates.date2num(chart_df["time"])
        x_min, x_max = times.min() - 0.5, times.max() + 0.5
        ax.set_xlim(x_min, x_max)

        # ── Session backgrounds ──
        session_colors = {
            "Asian": (0.1, 0.2, 0.4, 0.15),
            "London": (0.2, 0.4, 0.2, 0.12),
            "New_York": (0.4, 0.2, 0.2, 0.12),
        }
        for i in range(len(chart_df) - 1):
            t_start = times[i]
            t_end = times[i + 1]
            hour = chart_df["time"].iloc[i].hour
            if 0 <= hour < 9:
                color = session_colors["Asian"]
            elif 8 <= hour < 17:
                color = session_colors["London"]
            elif 13 <= hour < 22:
                color = session_colors["New_York"]
            else:
                continue
            ax.axvspan(t_start, t_end, facecolor=color, edgecolor="none")

        # ── Draw candlesticks ──
        candle_width = (times.max() - times.min()) / len(chart_df) * 0.6

        for i in range(len(chart_df)):
            row = chart_df.iloc[i]
            t = times[i]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]

            is_bull = c >= o
            body_color = "#00c853" if is_bull else "#ff1744"
            wick_color = "#4a4a6a"

            # Wick
            ax.plot([t, t], [l, h], color=wick_color, linewidth=1.2, zorder=1)

            # Body
            body_bottom = min(o, c)
            body_height = abs(c - o) or 0.00001
            rect = mpatches.Rectangle(
                (t - candle_width / 2, body_bottom),
                candle_width, body_height,
                facecolor=body_color,
                edgecolor=body_color,
                linewidth=0.5,
                zorder=2,
            )
            ax.add_patch(rect)

        # ── EMA20 line ──
        ema_valid = chart_df["ema20"].notna()
        if ema_valid.any():
            ax.plot(times[ema_valid], chart_df["ema20"][ema_valid],
                    color="#ffd700", linewidth=1.5, alpha=0.8,
                    label="EMA20", zorder=3)

        # ── Prediction markers ──
        if predictions:
            chart_start = chart_df["time"].iloc[0]
            chart_end = chart_df["time"].iloc[-1]

            for pred in predictions:
                try:
                    pred_time = pd.to_datetime(pred["time"])
                except Exception:
                    continue

                if pred_time < chart_start or pred_time > chart_end:
                    continue

                matches = chart_df[chart_df["time"] == pred_time]
                if matches.empty:
                    idx = (chart_df["time"] - pred_time).abs().idxmin()
                    if idx < 0 or idx >= len(chart_df):
                        continue
                    pred_price = chart_df.iloc[idx]["close"]
                    pred_x = times[idx]
                else:
                    pred_price = matches.iloc[0]["close"]
                    pred_x = mdates.date2num(pd.to_datetime(matches.iloc[0]["time"]))

                is_haut = pred["direction"] == "PLUS_HAUT"
                is_correct = pred.get("correct", False)
                arrow_color = "#00e676" if is_correct else "#ff5252"

                y_offset = y_range * 0.02
                if is_haut:
                    y_pos = pred_price - y_offset * 2
                    ax.scatter(pred_x, y_pos, marker="^", s=80,
                              color=arrow_color, edgecolors="white",
                              linewidth=0.5, zorder=5, alpha=0.9)
                else:
                    y_pos = pred_price + y_offset * 2
                    ax.scatter(pred_x, y_pos, marker="v", s=80,
                              color=arrow_color, edgecolors="white",
                              linewidth=0.5, zorder=5, alpha=0.9)

        # ── Formatting ──
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8, color="#8a8aaa")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.5f}"))
        ax.tick_params(axis="y", colors="#8a8aaa", labelsize=9)

        ax.set_title("EUR/USD — Bougies 1 Minute" + (" avec Prédictions" if predictions else ""),
                    color="#e0e0e0", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Heure (UTC)", color="#8a8aaa", fontsize=10)
        ax.set_ylabel("Prix", color="#8a8aaa", fontsize=10)

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor="#00c853", edgecolor="none", label="Haussière"),
            mpatches.Patch(facecolor="#ff1744", edgecolor="none", label="Baissière"),
            plt.Line2D([0], [0], color="#ffd700", linewidth=1.5, label="EMA20"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#00e676",
                       markersize=8, label="✅ Correcte"),
            plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#ff5252",
                       markersize=8, label="❌ Fausse"),
        ]
        ax.legend(handles=legend_elements, loc="upper left",
                 facecolor="#1a1a2e", edgecolor="#333", labelcolor="#e0e0e0",
                 fontsize=9)

        ax.grid(True, alpha=0.15, color="#4a4a6a", linestyle="--", linewidth=0.5)

        # Session labels
        text_y = y_max + padding * 0.8
        ax.text(times[len(times)//4], text_y, "🕐 Asian",
               color="#4a8aba", fontsize=8, ha="center", alpha=0.7)
        ax.text(times[len(times)//2], text_y, "🕐 London",
               color="#4aba4a", fontsize=8, ha="center", alpha=0.7)
        ax.text(times[3*len(times)//4], text_y, "🕐 New York",
               color="#ba4a4a", fontsize=8, ha="center", alpha=0.7)

        # Borders
        for spine in ax.spines.values():
            spine.set_color("#333")
            spine.set_linewidth(0.5)

        plt.tight_layout()
        _fig = fig  # Keep reference before cleanup
        fig = None  # Prevent finally from closing it
        return _fig

    except Exception as e:
        print(f"[!] Erreur graphique: {e}")
        return None
    finally:
        # Clean up figure only on error to prevent memory leak
        if fig is not None:
            plt.close(fig)


# ============================================================
# Gradio UI Functions
# ============================================================

def backtest_ui(days: int) -> Tuple[str, Optional[Figure]]:
    """Run backtest and return formatted results + candlestick chart."""
    n_candles = days * 60 * 24  # ~60 candles/hour * 24h
    n_candles = min(n_candles, 2000)

    output = f"## 📊 Backtest EUR/USD — {days} jours\n\n"
    output += f"Récupération des données yfinance...\n"

    df = fetch_historical_data(count=n_candles)
    if df.empty:
        return "❌ Impossible de récupérer les données yfinance. Veuillez réessayer.", None

    output += f"✅ {len(df)} bougies 1-minute chargées\n\n"

    predictions, stats = run_backtest(df)
    if not predictions:
        return "❌ Aucune prédiction générée (pas assez de données ou conditions non remplies).", None

    winrate = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0

    output += "### 📈 Résultats Globaux\n\n"
    output += f"| Métrique | Valeur |\n|----------|-------|\n"
    output += f"| Total prédictions | {stats['total']} |\n"
    output += f"| ✅ Correctes | {stats['correct']} |\n"
    output += f"| ❌ Fausses | {stats['wrong']} |\n"
    output += f"| **Winrate** | **{winrate:.1f}%** |\n"
    output += f"| Score moyen | {sum(stats['scores'])/len(stats['scores']):.1f} |\n"
    output += f"| Confiance moyenne | {sum(stats['confidences'])/len(stats['confidences']):.1f}% |\n\n"

    output += "### 🔄 Par Direction\n\n"
    haut_wr = stats["haut_correct"] / stats["plus_haut"] * 100 if stats["plus_haut"] > 0 else 0
    bas_wr = stats["bas_correct"] / stats["plus_bas"] * 100 if stats["plus_bas"] > 0 else 0
    output += f"| Direction | Total | Correct | Winrate |\n"
    output += f"|-----------|-------|---------|--------|\n"
    output += f"| 🟢 PLUS HAUT | {stats['plus_haut']} | {stats['haut_correct']} | {haut_wr:.1f}% |\n"
    output += f"| 🔴 PLUS BAS | {stats['plus_bas']} | {stats['bas_correct']} | {bas_wr:.1f}% |\n"

    output += "\n### 🕐 Par Session\n\n"
    for sess in ["Asian", "London", "New_York"]:
        s = stats.get("sessions", {}).get(sess, {"ok": 0, "total": 0})
        if s["total"] > 0:
            wr = s["ok"] / s["total"] * 100
            bar = "█" * int(wr / 10) + "░" * (10 - int(wr / 10))
            output += f"| {sess:<12} | {bar} {wr:.1f}% ({s['ok']}/{s['total']}) |\n"

    output += "\n### 📋 Dernières 15 Prédictions\n\n"
    output += "| # | Heure | Prix | Direction | Score | Confiance | Résultat | Pips |\n"
    output += "|---|-------|------|-----------|-------|-----------|----------|------|\n"
    for i, p in enumerate(predictions[-15:], 1):
        icon = "✅" if p["correct"] else "❌"
        dir_icon = "🟢" if p["direction"] == "PLUS_HAUT" else "🔴"
        output += f"| {i} | {str(p['time'])[-8:]} | {p['price']:.5f} | {dir_icon} {p['direction']:<10} | {p['score']:+d} | {p['confidence']}% | {icon} | {p['change_pips']:+.1f} |\n"

    # Generate candlestick chart
    fig = generate_candlestick_chart(df, predictions, max_candles=80)

    return output, fig


def live_analysis_ui() -> str:
    """Get current market analysis from yfinance."""
    output = f"## 🔍 Analyse Temps Réel EUR/USD\n\n"
    output += f"*Données yfinance (15 min de retard - compte gratuit)*\n\n"

    df = fetch_historical_data(count=200)
    if df.empty:
        return "❌ Impossible de récupérer les données."

    momentum = MomentumAnalyzer()
    liquidity = LiquidityAnalyzer()

    current_price = float(df["close"].iloc[-1])
    ema_val = momentum.ema(df)
    rsi_val = momentum.rsi(df)
    atr_val = momentum.atr(df)
    zones = liquidity.find_swing_highs_lows(df)

    output += f"### 💰 Prix Actuel\n\n"
    output += f"**{current_price:.5f}** USD\n\n"

    output += "### 📊 Indicateurs Techniques\n\n"
    output += "| Indicateur | Valeur | Signal |\n|------------|-------|--------|\n"

    # EMA signal
    ema_signal = "🟢 HAUSSIER" if current_price > ema_val else "🔴 BAISSIER"
    output += f"| EMA20 | {ema_val:.5f} | {ema_signal} |\n"

    # RSI signal
    if rsi_val > 70:
        rsi_signal = "⚠️ Suracheté"
    elif rsi_val < 30:
        rsi_signal = "⚠️ Survente"
    elif rsi_val > 50:
        rsi_signal = "🟢 HAUSSIER"
    else:
        rsi_signal = "🔴 BAISSIER"
    output += f"| RSI7 | {rsi_val:.1f} | {rsi_signal} |\n"

    output += f"| ATR14 | {atr_val:.5f} | Volatilité {'haute' if atr_val > 0.0003 else 'basse'} |\n"

    output += "\n### 🏔️ Zones de Liquidité\n\n"
    output += f"| Zone | Niveau |\n|------|-------|\n"
    output += f"| Plus haut (48) | {zones['swing_high']:.5f} |\n"
    output += f"| Plus bas (48) | {zones['swing_low']:.5f} |\n\n"

    # Run prediction
    engine = PredictionEngine()
    pred = engine.predict_2min(df)
    if pred:
        output += f"### 🎯 Prédiction 2 minutes\n\n"
        dir_icon = "🟢" if pred["direction"] == "PLUS_HAUT" else "🔴"
        output += f"**{dir_icon} {pred['direction']}** | Score: {pred['score']:+d} | Confiance: {pred['confidence']}%\n\n"
        output += "Détails:\n"
        for d in pred.get("details", []):
            output += f"- {d['icon']} {d['label']}: {d['value']}\n"
    else:
        output += "*Aucune prédiction pour l'instant (conditions non remplies)*\n"

    return output


def memory_analysis_ui() -> str:
    """Analyze memory.json performance."""
    history, stats = load_memory_data()
    if not history:
        return "❌ Aucune donnée mémoire trouvée. Lancez le bot pour collecter des données !"

    output = f"## 💾 Analyse de Performance — memory.json\n\n"
    output += f"**{stats['total']} entrées** | "
    output += f"Winrate: **{stats['winrate']}%**\n\n"

    output += "### 📊 Synthèse\n\n"
    output += f"| Métrique | Valeur |\n|----------|-------|\n"
    output += f"| Total entrées | {stats['total']} |\n"
    output += f"| ✅ Correctes | {stats['correct']} |\n"
    output += f"| ❌ Fausses | {stats['wrong']} |\n"
    output += f"| **Winrate** | **{stats['winrate']}%** |\n\n"

    # Par session
    output += "### 🕐 Par Session\n\n"
    output += "| Session | Winrate |\n|---------|--------|\n"
    for sess in ["Asian", "London", "New_York"]:
        s = stats.get("sessions", {}).get(sess, {"ok": 0, "total": 0})
        if s["total"] > 0:
            wr = s["ok"] / s["total"] * 100
            bar = "█" * int(wr / 10) + "░" * (10 - int(wr / 10))
            output += f"| {sess} | {bar} {wr:.1f}% ({s['ok']}/{s['total']}) |\n"

    # Par direction
    output += "\n### 🎯 Par Direction\n\n"
    output += "| Direction | Winrate |\n|----------|--------|\n"
    for d in ["PLUS_HAUT", "PLUS_BAS"]:
        s = stats.get("directions", {}).get(d, {"ok": 0, "total": 0})
        if s["total"] > 0:
            wr = s["ok"] / s["total"] * 100
            label = "🟢 HAUT" if d == "PLUS_HAUT" else "🔴 BAS"
            output += f"| {label} | {wr:.1f}% ({s['ok']}/{s['total']}) |\n"

    # Top profiles
    output += "\n### 🔥 Top Profils (pattern|session)\n\n"
    output += "| Profil | Winrate |\n|--------|--------|\n"
    profiles = stats.get("profiles", {})
    sorted_profiles = sorted(profiles.items(), key=lambda x: -x[1]["total"])[:10]
    for prof, s in sorted_profiles:
        if s["total"] > 0:
            wr = s["ok"] / s["total"] * 100
            bar = "█" * int(wr / 10) + "░" * (10 - int(wr / 10))
            output += f"| `{prof}` | {bar} {wr:.1f}% ({s['ok']}/{s['total']}) |\n"

    # Évolution
    output += "\n### 📈 Évolution (blocs de 20)\n\n"
    block_size = 20
    for i in range(0, stats["total"], block_size):
        block = history[i:i + block_size]
        ok = sum(1 for e in block if e["outcome"])
        wr = ok / len(block) * 100
        bar = "█" * int(wr / 10) + "░" * (10 - int(wr / 10))
        output += f"| #{i//block_size + 1} ({i}-{i+len(block)-1}) | {bar} {wr:.1f}% ({ok}/{len(block)}) |\n"

    return output


def ml_training_ui() -> str:
    """Train ML model and show results."""
    history, stats = load_memory_data()
    if not history:
        return "❌ Aucune donnée mémoire. Impossible d'entraîner le ML."

    output = f"## 🤖 Entraînement du Modèle ML\n\n"
    output += f"**{len(history)} échantillons disponibles**\n\n"

    if not HAS_SKLEARN:
        return "❌ scikit-learn n'est pas installé sur ce Space."

    output += "Lancement de l'entraînement RandomForest...\n\n"

    result = train_ml_on_memory(history)
    if result is None:
        return "❌ L'entraînement a échoué."

    if result["status"] == "not_enough_data":
        return f"⚠️ Pas assez de données ({result['samples']} échantillons). Minimum requis: 10."

    output += "### ✅ Modèle Entraîné avec Succès\n\n"
    output += f"**Échantillons:** {result['samples']}\n"
    output += f"**Algorithme:** RandomForest (100 arbres, profondeur max=5)\n\n"

    output += "### 🔮 Tests de Prédiction (ML Override)\n\n"
    output += "Quand le ML est confiant (≥60%), il peut remplacer la direction de la stratégie.\n\n"
    output += "| # | Outcome réel | ML Override | Direction ML |\n"
    output += "|---|-------------|-------------|-------------|\n"
    for i, t in enumerate(result.get("test_results", [])[:10], 1):
        outcome_icon = "✅" if t["actual_outcome"] else "❌"
        override_icon = "🔄" if t["ml_override"] else "➖"
        output += f"| {i} | {outcome_icon} | {override_icon} | {t['ml_direction']} |\n"

    return output


def strategy_explainer_ui() -> str:
    """Explain the bot's strategy in detail."""
    return """
## 📚 Explication de la Stratégie

### Architecture Globale

```
Données M1 (yfinance/MT5/Dukascopy)
        ↓
    CandleBuilder (bougies 1-min)
        ↓
    ┌──────────────────────────────┐
    │  Stratégie 1: Comptage       │
    │  Bougies Consecutives        │
    │  + Momentum + Session        │
    └──────────┬───────────────────┘
              ↓
    ┌──────────────────────────────┐
    │  Stratégie 2: Confirmation   │
    │  EMA20 + RSI7 + Filtre ATR  │
    └──────────┬───────────────────┘
              ↓ (accord des 2)
        Prédiction T+2
        ↓           ↓
    Discord    Vérification
    Webhook    à T+2
                  ↓
             memory.json
                  ↓
            ML RandomForest
            (apprentissage)
```

### 🔵 Stratégie 1 — Comptage Bougies

1. **3 bougies vertes consécutives** → Signal **PLUS_HAUT**
2. **3 bougies rouges consécutives** → Signal **PLUS_BAS**
3. **Score basé sur:** nombre de bougies, momentum, accélération, liquidité, session

### 🔴 Stratégie 2 — Confirmation Tendance (EMA20 + RSI7)

Valide le signal avec des indicateurs techniques :

| Condition | PLUS_HAUT | PLUS_BAS |
|-----------|-----------|----------|
| Prix vs EMA20 | Prix > EMA20 ✅ | Prix < EMA20 ✅ |
| RSI7 | RSI > 50 ✅ | RSI < 50 ✅ |
| ATR minimum | > 0.00005 ✅ | > 0.00005 ✅ |

**Les 2 stratégies doivent être d'accord** pour qu'un signal soit émis.

### 🤖 Machine Learning (RandomForest)

Après chaque vérification T+2, le résultat est enregistré dans `memory.json`.

Le modèle RandomForest apprend à prédire la direction réelle du prix avec les features :
- `green_count`, `red_count` — nombre de bougies vertes/rouges
- `score` — score de la stratégie
- `atr` — volatilité actuelle
- `rsi` — RSI 7 périodes
- `price_ema_diff` — distance au EMA20
- `hour`, `session`, `overlap` — contexte temporel

**Override ML**: Si la confiance du modèle ≥ 60%, il peut remplacer la direction.

### 💡 Exemple de Flow

```
Bougies: 🟢🟢🟢 → Signal HAUT (score: +6)
EMA20: Prix au-dessus ✅
RSI7: 62.3 > 50 ✅
→ ✅ Prédiction: PLUS_HAUT dans 2 min
→ Envoi Discord
→ Attente 2 min
→ Vérification: Prix a monté ? ✅ (enregistré)
→ ML: entraîné tous les 10 échantillons
```
"""


def discord_preview_ui() -> str:
    """Preview how Discord messages look."""
    return """
## 💬 Aperçu Messages Discord

### 📤 Prédiction Envoyée
```
🟢 **EUR/USD — PLUS HAUT dans 2 min**

Prix: 1.10452 | Score: +6
EMA20: 1.10410 | RSI7: 62.3
ATR: 0.00040 | Confiance: 86%

**Analyse:**
📊 Bougies: 3 vertes / 0 rouges _+6_
⚡ Momentum (RSI7): RSI 62.3 > 55 +2
🏔️ Structure: Cassure sommet +2
🕐 Session: London

━━━━━━━━━━━━━━━━━━
```

### ✅ Vérification
```
✅ **Verification prediction — CORRECT**

Prediction: PLUS HAUT dans 2 min
Prix T+0: 1.10452
Prix T+2: 1.10500
📈 Variation: +4.8 pips

━━━━━━━━━━━━━━━━━━
```

### ❌ Vérification Échouée
```
❌ **Verification prediction — FAUX**

Prediction: PLUS BAS dans 2 min
Prix T+0: 1.10452
Prix T+2: 1.10520
📈 Variation: +6.8 pips

━━━━━━━━━━━━━━━━━━
```

### 🔄 ML Override
```
ML override: 3g0r → PLUS_BAS (proba=72%)
```
Le modèle ML a détecté que malgré 3 bougies vertes,
le contexte (RSI, heure, patterns historiques) suggère
un retournement baissier.
"""


# ============================================================
# UI Styling
# ============================================================

CSS = """
<style>
    .container { max-width: 1200px; margin: auto; }
    .header { text-align: center; padding: 2rem 0; }
    .footer { text-align: center; padding: 2rem 0; color: #888; }
    .output-box { border-left: 4px solid #4CAF50; padding: 1rem; background: #f9f9f9; border-radius: 0 8px 8px 0; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #f0f0f0; font-weight: 600; }
    code { background: #e8e8e8; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
</style>
"""

# ============================================================
# Main App
# ============================================================

def build_app():
    """Build the Gradio interface."""

    with gr.Blocks(
        title="Forex Algo Bot - Dashboard",
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="blue",
        ),
        css=CSS
    ) as app:

        # Header
        gr.Markdown("""
        # 📈 **Forex Algo Bot** — EUR/USD M1

        *Stratégie combinée: Comptage bougies + EMA20/RSI7 + ML RandomForest*
        """)

        with gr.Row():
            gr.Markdown(f"""
            <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1rem;">
                <span>🤖 **Bot OK:** {'✅' if BOT_OK else '❌'}</span>
                <span>🧠 **scikit-learn:** {'✅' if HAS_SKLEARN else '❌'}</span>
                <span>📊 **Data:** yfinance (gratuit, ~15min retard)</span>
            </div>
            """)

        with gr.Tabs():
            # Tab 1: Live Analysis
            with gr.TabItem("🔍 Analyse Temps Réel", id=0):
                with gr.Row():
                    live_btn = gr.Button("🔄 Actualiser l'Analyse", variant="primary", size="lg", scale=2)
                with gr.Row():
                    live_output = gr.Markdown("Cliquez pour analyser le marché EUR/USD en direct...")
                live_btn.click(fn=live_analysis_ui, inputs=[], outputs=live_output)

            # Tab 2: Backtest
            with gr.TabItem("📊 Backtest", id=1):
                with gr.Row():
                    days_slider = gr.Slider(
                        minimum=1, maximum=5, value=2, step=1,
                        label="Nombre de jours à backtester",
                        info="Plus de jours = plus de données mais temps d'exécution plus long",
                    )
                with gr.Row():
                    backtest_btn = gr.Button("🚀 Lancer le Backtest", variant="primary", size="lg")
                with gr.Row():
                    backtest_output = gr.Markdown("Configurez et lancez un backtest...")
                with gr.Row():
                    backtest_chart = gr.Plot(label="Graphique Bougies avec Prédictions")
                backtest_btn.click(
                    fn=backtest_ui,
                    inputs=[days_slider],
                    outputs=[backtest_output, backtest_chart],
                )

            # Tab 3: Memory Analysis
            with gr.TabItem("💾 Analyse Mémoire", id=2):
                with gr.Row():
                    memory_btn = gr.Button("📊 Analyser memory.json", variant="primary", size="lg")
                with gr.Row():
                    memory_output = gr.Markdown(
                        "Analysez les performances historiques du bot...\n\n"
                        "*Les données viennent de memory.json*"
                    )
                memory_btn.click(fn=memory_analysis_ui, inputs=[], outputs=memory_output)

            # Tab 4: ML Training
            with gr.TabItem("🤖 Apprentissage ML", id=3):
                with gr.Row():
                    gr.Markdown("""
                    ### 🤖 RandomForest Classifier

                    Entraîne le modèle ML sur les données de `memory.json` pour prédire
                    la direction du prix et potentiellement override la stratégie.

                    **Features:** green_count, red_count, score, atr, rsi, price_ema_diff, hour, session, overlap
                    """)
                with gr.Row():
                    ml_btn = gr.Button("🧠 Entraîner le Modèle", variant="primary", size="lg")
                with gr.Row():
                    ml_output = gr.Markdown("Cliquez pour entraîner le modèle ML...")
                ml_btn.click(fn=ml_training_ui, inputs=[], outputs=ml_output)

            # Tab 5: Strategy Explain
            with gr.TabItem("📚 Stratégie", id=4):
                strategy_output = gr.Markdown(strategy_explainer_ui())

            # Tab 6: Discord Preview
            with gr.TabItem("💬 Discord", id=5):
                discord_output = gr.Markdown(discord_preview_ui())

        # Footer
        gr.Markdown("""
        ---
        <div style="text-align:center; padding:1rem; color:#666;">
            <p>
                📈 <strong>Forex Algo Bot</strong> — EUR/USD M1 |
                Stratégie: Bougies + EMA20/RSI7 + ML RandomForest |
                Propulsé par yfinance, scikit-learn & Gradio |
                <a href="https://huggingface.co" target="_blank">Hugging Face Spaces</a>
            </p>
        </div>
        """)

    return app


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  FOREX ALGO BOT — Hugging Face Space")
    print("  Backtest, visualisation & analyse de performance")
    print("=" * 60)
    print(f"\n  Bot modules: {'OK' if BOT_OK else 'NON DISPONIBLE'}")
    print(f"  scikit-learn: {'OK' if HAS_SKLEARN else 'NON DISPONIBLE'}")
    print(f"  yfinance: disponible")
    print()

    app = build_app()
    
    # Port pour Render.com / Hugging Face / services cloud
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)
