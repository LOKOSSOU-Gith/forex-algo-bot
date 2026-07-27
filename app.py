#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║     FOREX ALGO BOT — Dashboard de Performance                     ║
║     Backtest, visualisation & analyse temps réel                   ║
║     Stratégie: comptage bougies + EMA20 + RSI7 sur EUR/USD M1     ║
╚══════════════════════════════════════════════════════════════════════╝

Déployable sur Replit, Render.com ou Hugging Face Spaces.
Utilise yfinance (gratuit) pour les données historiques et
scikit-learn pour les modèles ML.
"""

import os
import sys
import json
import io
import math
import warnings
import subprocess
import signal
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ─── Matplotlib ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    plt = Figure = None

import gradio as gr

# ─── Path ──────────────────────────────────────────────────────────────────
_BOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(_BOT_DIR))

# ─── Imports du bot ────────────────────────────────────────────────────────
try:
    from strategy.engine import PredictionEngine
    from strategy.momentum import MomentumAnalyzer
    from strategy.liquidity import LiquidityAnalyzer
    from learning.model import MLModel, HAS_SKLEARN
    BOT_OK = True
except Exception as e:
    BOT_OK = False
    print(f"[!] Erreur import bot: {e}")

SYMBOL = "EURUSD"
LOOKBACK = 500

# ─── Bot process control ────────────────────────────────────────────────────
_bot_process: Optional[subprocess.Popen] = None
_bot_start_time: Optional[float] = None

# ====================================================================
# DATA LOADING (identique à l'original)
# ====================================================================

def fetch_historical_data(symbol: str = SYMBOL, count: int = 500) -> pd.DataFrame:
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
        df = df.reset_index()
        time_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else None)
        if time_col:
            df["time"] = df[time_col]
        df = df.drop(columns=["Datetime", "Date"], errors="ignore")
        if hasattr(df["time"].dtype, "tz") and df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_localize(None)
        return df.tail(count).reset_index(drop=True)
    except Exception as e:
        print(f"[!] Erreur yfinance: {e}")
        return pd.DataFrame()


def run_backtest(df: pd.DataFrame) -> Tuple[List[Dict], Dict]:
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
        "total": total, "correct": correct, "wrong": total - correct,
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
    if not HAS_SKLEARN or not history:
        return None
    model = MLModel(model_path=str(_BOT_DIR / "model.pkl"), min_samples=10)
    trained = model.train(history)
    if not trained:
        return {"status": "not_enough_data", "samples": len(history)}
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
            "current_price": 1.10, "ema": 1.10,
            "timestamp": datetime.now(),
        }
        override = model.should_override(pred)
        test_results.append({
            "actual_outcome": entry["outcome"],
            "ml_override": override is not None,
            "ml_direction": override or "none",
        })
    return {
        "status": "trained", "samples": len(history),
        "test_results": test_results[:10],
        "feature_importances": model.model.feature_importances_.tolist() if model.model is not None else [],
        "features_names": model.FEATURE_NAMES if model.model is not None else [],
    }


# ====================================================================
# CHARTS (NOUVEAU — graphiques améliorés)
# ====================================================================

def generate_equity_curve(predictions: List[Dict]) -> Optional[Figure]:
    """Equity curve based on backtest predictions (1 pip = hypothetical $1)."""
    if not MATPLOTLIB_OK or not predictions:
        return None
    try:
        cumulative = 0
        equity = []
        times = []
        for p in predictions:
            cumulative += p["change_pips"] if p["correct"] else -abs(p["change_pips"])
            equity.append(cumulative)
            try:
                times.append(pd.to_datetime(p["time"]))
            except Exception:
                times.append(len(equity))

        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        x = times if isinstance(times[0], (pd.Timestamp, datetime)) else list(range(len(equity)))
        ax.fill_between(range(len(equity)), equity, alpha=0.3,
                        color="#00c853" if equity[-1] >= 0 else "#ff1744")
        ax.plot(range(len(equity)), equity, color="#00e676" if equity[-1] >= 0 else "#ff5252",
                linewidth=1.5, zorder=3)

        ax.axhline(y=0, color="#4a4a6a", linewidth=0.8, linestyle="--")
        ax.set_title("💰 Simulation P&L (1 pip = 1$)", color="#e0e0e0",
                     fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Trade #", color="#8a8aaa", fontsize=8)
        ax.set_ylabel("P&L ($)", color="#8a8aaa", fontsize=8)
        ax.tick_params(colors="#8a8aaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.grid(True, alpha=0.1, color="#4a4a6a", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"[!] Erreur equity curve: {e}")
        return None


def generate_session_chart(stats: Dict) -> Optional[Figure]:
    """Bar chart: winrate per session."""
    if not MATPLOTLIB_OK:
        return None
    try:
        sessions = ["Asian", "London", "New_York"]
        winrates = []
        totals = []
        colors = []
        for s in sessions:
            data = stats.get("sessions", {}).get(s, {"ok": 0, "total": 0})
            wr = (data["ok"] / data["total"] * 100) if data["total"] > 0 else 0
            winrates.append(wr)
            totals.append(data["total"])
            colors.append("#00c853" if wr >= 50 else "#ff1744")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        bars = ax.bar(sessions, winrates, color=colors, alpha=0.85,
                      edgecolor="#4a4a6a", linewidth=0.5, width=0.5)
        for bar, wr, total in zip(bars, winrates, totals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{wr:.1f}%\n({total} trades)", ha="center", va="bottom",
                    color="#e0e0e0", fontsize=9, fontweight="bold")

        ax.axhline(y=50, color="#ffd700", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_ylim(0, max(max(winrates) + 15, 65))
        ax.set_title("🎯 Winrate par Session", color="#e0e0e0",
                     fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Winrate (%)", color="#8a8aaa", fontsize=9)
        ax.tick_params(colors="#8a8aaa", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.grid(True, alpha=0.1, color="#4a4a6a", linestyle="--", linewidth=0.5, axis="y")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"[!] Erreur session chart: {e}")
        return None


def generate_feature_importance_chart(importances: List[float], feature_names: List[str]) -> Optional[Figure]:
    """Horizontal bar chart of ML feature importances."""
    if not MATPLOTLIB_OK or not importances or not feature_names:
        return None
    try:
        pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        names = [p[0] for p in pairs]
        values = [p[1] for p in pairs]

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        bars = ax.barh(range(len(names)), values, color=colors, alpha=0.85,
                       edgecolor="#4a4a6a", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, color="#e0e0e0", fontsize=9)
        ax.invert_yaxis()

        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.1%}", ha="left", va="center", color="#8a8aaa", fontsize=8)

        ax.set_title("🔬 Importance des Features ML", color="#e0e0e0",
                     fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Importance relative", color="#8a8aaa", fontsize=9)
        ax.tick_params(colors="#8a8aaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.grid(True, alpha=0.1, color="#4a4a6a", linestyle="--", linewidth=0.5, axis="x")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"[!] Erreur feature importance: {e}")
        return None


def generate_evolution_chart(history: List[Dict], block_size: int = 20) -> Optional[Figure]:
    """Evolution of winrate over time (rolling blocks)."""
    if not MATPLOTLIB_OK or not history:
        return None
    try:
        blocks = []
        winrates = []
        for i in range(0, len(history), block_size):
            block = history[i:i + block_size]
            ok = sum(1 for e in block if e["outcome"])
            wr = ok / len(block) * 100
            blocks.append(f"#{i//block_size + 1}")
            winrates.append(wr)

        fig, ax = plt.subplots(figsize=(9, 3))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        colors = ["#00c853" if w >= 50 else "#ff1744" for w in winrates]
        ax.bar(blocks, winrates, color=colors, alpha=0.8, edgecolor="#4a4a6a", linewidth=0.5)

        ax.axhline(y=50, color="#ffd700", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_ylim(0, 100)
        ax.set_title("📈 Évolution du Winrate (blocs de {})".format(block_size),
                     color="#e0e0e0", fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Winrate (%)", color="#8a8aaa", fontsize=9)
        ax.tick_params(colors="#8a8aaa", labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.grid(True, alpha=0.1, color="#4a4a6a", linestyle="--", linewidth=0.5, axis="y")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"[!] Erreur evolution chart: {e}")
        return None


def generate_hourly_heatmap(stats: Dict) -> Optional[Figure]:
    """Heatmap-like chart: winrate by hour."""
    if not MATPLOTLIB_OK:
        return None
    try:
        hours = stats.get("hours", {})
        if not hours:
            return None

        data = {int(h): v for h, v in hours.items() if v["total"] > 0}
        if not data:
            return None

        all_hours = sorted(data.keys())
        winrates = [data[h]["ok"] / data[h]["total"] * 100 for h in all_hours]
        totals = [data[h]["total"] for h in all_hours]

        fig, ax = plt.subplots(figsize=(10, 2.5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        colors = ["#00c853" if w >= 50 else "#ff1744" for w in winrates]
        bars = ax.bar([str(h) + "h" for h in all_hours], winrates, color=colors, alpha=0.85,
                      edgecolor="#4a4a6a", linewidth=0.5, width=0.6)

        for bar, wr, total in zip(bars, winrates, totals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{wr:.0f}%", ha="center", va="bottom",
                    color="#e0e0e0", fontsize=7, fontweight="bold")

        ax.axhline(y=50, color="#ffd700", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_ylim(0, max(max(winrates) + 12, 65))
        ax.set_title("🕐 Winrate par Heure (UTC)", color="#e0e0e0",
                     fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Winrate (%)", color="#8a8aaa", fontsize=8)
        ax.tick_params(colors="#8a8aaa", labelsize=7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.grid(True, alpha=0.1, color="#4a4a6a", linestyle="--", linewidth=0.5, axis="y")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"[!] Erreur hourly chart: {e}")
        return None


# ====================================================================
# CANDLESTICK CHART (amélioré)
# ====================================================================

def generate_candlestick_chart(df: pd.DataFrame,
                                predictions: Optional[List[Dict]] = None,
                                max_candles: int = 80) -> Optional[Figure]:
    if not MATPLOTLIB_OK or df.empty:
        return None
    fig = None
    try:
        chart_df = df.tail(max_candles).copy()
        if len(chart_df) < 10:
            return None
        if not np.issubdtype(chart_df["time"].dtype, np.datetime64):
            chart_df["time"] = pd.to_datetime(chart_df["time"])

        closes = chart_df["close"].values
        chart_df["ema20"] = pd.Series(closes).ewm(span=20, adjust=False).mean()
        chart_df["bb_mid"] = chart_df["ema20"]
        chart_df["bb_std"] = closes[-len(chart_df):].std() if len(closes) > 20 else 0.0001
        chart_df["bb_upper"] = chart_df["ema20"] + 2 * chart_df["bb_std"]
        chart_df["bb_lower"] = chart_df["ema20"] - 2 * chart_df["bb_std"]

        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        y_min = chart_df["low"].min()
        y_max = chart_df["high"].max()
        y_range = y_max - y_min
        padding = y_range * 0.1
        ax.set_ylim(y_min - padding, y_max + padding)

        times = mdates.date2num(chart_df["time"])
        x_min, x_max = times.min() - 0.5, times.max() + 0.5
        ax.set_xlim(x_min, x_max)

        # Session backgrounds
        session_colors = {
            "Asian": (0.1, 0.2, 0.4, 0.12),
            "London": (0.2, 0.4, 0.2, 0.10),
            "New_York": (0.4, 0.2, 0.2, 0.10),
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

        # Bollinger Bands
        bb_valid = chart_df["bb_upper"].notna()
        if bb_valid.any():
            ax.fill_between(times[bb_valid], chart_df["bb_upper"][bb_valid],
                            chart_df["bb_lower"][bb_valid],
                            alpha=0.08, color="#4a8aba", label="Bollinger ±2σ")

        # Candlesticks
        candle_width = (times.max() - times.min()) / len(chart_df) * 0.6
        for i in range(len(chart_df)):
            row = chart_df.iloc[i]
            t = times[i]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            is_bull = c >= o
            body_color = "#00c853" if is_bull else "#ff1744"
            wick_color = "#4a4a6a"
            ax.plot([t, t], [l, h], color=wick_color, linewidth=1, zorder=1)
            body_bottom = min(o, c)
            body_height = abs(c - o) or 0.00001
            rect = mpatches.Rectangle(
                (t - candle_width / 2, body_bottom),
                candle_width, body_height,
                facecolor=body_color, edgecolor=body_color,
                linewidth=0.3, zorder=2,
            )
            ax.add_patch(rect)

        # EMA20
        ema_valid = chart_df["ema20"].notna()
        if ema_valid.any():
            ax.plot(times[ema_valid], chart_df["ema20"][ema_valid],
                    color="#ffd700", linewidth=1.5, alpha=0.8,
                    label="EMA20", zorder=3)

        # Predictions
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
                    ax.scatter(pred_x, pred_price - y_offset * 2, marker="^", s=80,
                              color=arrow_color, edgecolors="white", linewidth=0.5, zorder=5, alpha=0.9)
                else:
                    ax.scatter(pred_x, pred_price + y_offset * 2, marker="v", s=80,
                              color=arrow_color, edgecolors="white", linewidth=0.5, zorder=5, alpha=0.9)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8, color="#8a8aaa")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.5f}"))
        ax.tick_params(axis="y", colors="#8a8aaa", labelsize=9)

        ax.set_title("EUR/USD — Bougies 1 Minute" + (" avec Prédictions" if predictions else ""),
                    color="#e0e0e0", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Heure (UTC)", color="#8a8aaa", fontsize=10)
        ax.set_ylabel("Prix", color="#8a8aaa", fontsize=10)

        legend_elements = [
            mpatches.Patch(facecolor="#00c853", edgecolor="none", label="Haussière"),
            mpatches.Patch(facecolor="#ff1744", edgecolor="none", label="Baissière"),
            plt.Line2D([0], [0], color="#ffd700", linewidth=1.5, label="EMA20"),
            plt.Line2D([0], [0], color="#4a8aba", linewidth=4, alpha=0.3, label="Bollinger ±2σ"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#00e676",
                       markersize=8, label="✅ Correcte"),
            plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#ff5252",
                       markersize=8, label="❌ Fausse"),
        ]
        ax.legend(handles=legend_elements, loc="upper left",
                 facecolor="#1a1a2e", edgecolor="#333", labelcolor="#e0e0e0", fontsize=9)
        ax.grid(True, alpha=0.1, color="#4a4a6a", linestyle="--", linewidth=0.5)

        text_y = y_max + padding * 0.8
        ax.text(times[len(times)//4], text_y, "🕐 Asian",
               color="#4a8aba", fontsize=8, ha="center", alpha=0.7)
        ax.text(times[len(times)//2], text_y, "🕐 London",
               color="#4aba4a", fontsize=8, ha="center", alpha=0.7)
        ax.text(times[3*len(times)//4], text_y, "🕐 New York",
               color="#ba4a4a", fontsize=8, ha="center", alpha=0.7)

        for spine in ax.spines.values():
            spine.set_color("#333")

        plt.tight_layout()
        _fig = fig
        fig = None
        return _fig
    except Exception as e:
        print(f"[!] Erreur graphique: {e}")
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def generate_live_price_chart(df: pd.DataFrame) -> Optional[Figure]:
    """Live analysis price chart with EMA20 and RSI below."""
    if not MATPLOTLIB_OK or df.empty or len(df) < 30:
        return None
    try:
        chart_df = df.tail(120).copy()
        if not np.issubdtype(chart_df["time"].dtype, np.datetime64):
            chart_df["time"] = pd.to_datetime(chart_df["time"])

        closes = chart_df["close"].values
        chart_df["ema20"] = pd.Series(closes).ewm(span=20, adjust=False).mean()

        # RSI
        delta = np.diff(closes)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gains).rolling(14).mean().values
        avg_loss = pd.Series(losses).rolling(14).mean().values
        rs = np.where(avg_loss == 0, 100, avg_gain / np.where(avg_loss == 0, 1, avg_loss))
        rsi_values = 100 - (100 / (1 + rs))
        rsi_values = np.concatenate([[50], rsi_values])  # align length

        times = mdates.date2num(chart_df["time"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_facecolor("#0d1117")

        # Price panel
        ax1.set_facecolor("#161b22")
        ax1.plot(times, chart_df["close"], color="#58a6ff", linewidth=1.2, label="Prix", zorder=2)
        ax1.plot(times, chart_df["ema20"], color="#ffd700", linewidth=1, alpha=0.7, label="EMA20", zorder=3)
        ax1.fill_between(times, chart_df["close"], chart_df["ema20"],
                         where=(chart_df["close"] > chart_df["ema20"]),
                         color="#00c853", alpha=0.08, label="> EMA")
        ax1.fill_between(times, chart_df["close"], chart_df["ema20"],
                         where=(chart_df["close"] <= chart_df["ema20"]),
                         color="#ff1744", alpha=0.08, label="< EMA")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax1.set_title("EUR/USD — Live Price", color="#e0e0e0", fontsize=12, fontweight="bold", pad=10)
        ax1.set_ylabel("Prix", color="#8a8aaa", fontsize=9)
        ax1.tick_params(colors="#8a8aaa", labelsize=8)
        ax1.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#e0e0e0", fontsize=8)
        ax1.grid(True, alpha=0.1, color="#4a4a6a", linestyle="--", linewidth=0.5)
        for spine in ax1.spines.values():
            spine.set_color("#333")

        # RSI panel
        ax2.set_facecolor("#161b22")
        ax2.plot(times, rsi_values[:len(times)], color="#da70d6", linewidth=1.2, label="RSI14")
        ax2.axhline(y=70, color="#ff1744", linewidth=0.8, linestyle="--", alpha=0.5)
        ax2.axhline(y=30, color="#00c853", linewidth=0.8, linestyle="--", alpha=0.5)
        ax2.axhline(y=50, color="#4a4a6a", linewidth=0.5, linestyle="--", alpha=0.3)
        ax2.fill_between(times, 70, 100, color="#ff1744", alpha=0.05)
        ax2.fill_between(times, 0, 30, color="#00c853", alpha=0.05)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI", color="#8a8aaa", fontsize=9)
        ax2.set_xlabel("Heure (UTC)", color="#8a8aaa", fontsize=9)
        ax2.tick_params(colors="#8a8aaa", labelsize=8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax2.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#e0e0e0", fontsize=8)
        ax2.grid(True, alpha=0.1, color="#4a4a6a", linestyle="--", linewidth=0.5)
        for spine in ax2.spines.values():
            spine.set_color("#333")

        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"[!] Erreur live chart: {e}")
        return None


# ====================================================================
# GRADIO UI FUNCTIONS
# ====================================================================

def overview_ui() -> Tuple[str, Optional[Figure], Optional[Figure]]:
    """Accueil : KPIs globaux + live chart + sessions."""
    output = "## 📊 Vue d'Ensemble\n\n"
    df = fetch_historical_data(count=200)
    if df.empty:
        return "❌ Pas de données.", None, None

    momentum = MomentumAnalyzer()
    liquidity = LiquidityAnalyzer()
    current_price = float(df["close"].iloc[-1])
    ema_val = momentum.ema(df)
    rsi_val = momentum.rsi(df)
    atr_val = momentum.atr(df)

    # KPI badges
    ema_signal = "🟢 HAUSSIER" if current_price > ema_val else "🔴 BAISSIER"
    rsi_signal = "⚠️ Suracheté" if rsi_val > 70 else ("⚠️ Survente" if rsi_val < 30 else
                 ("🟢 HAUSSIER" if rsi_val > 50 else "🔴 BAISSIER"))
    atr_signal = "Haute" if atr_val > 0.0003 else "Basse"

    output += f"""
<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin:15px 0;">
  <div style="background:linear-gradient(135deg,#0d1117,#161b22); border:1px solid #333; border-radius:12px; padding:16px; text-align:center;">
    <div style="font-size:11px; color:#8a8aaa; text-transform:uppercase; letter-spacing:1px;">Prix</div>
    <div style="font-size:28px; font-weight:700; color:#58a6ff; margin:4px 0;">{current_price:.5f}</div>
    <div style="font-size:11px; color:#8a8aaa;">EUR/USD</div>
  </div>
  <div style="background:linear-gradient(135deg,#0d1117,#161b22); border:1px solid #333; border-radius:12px; padding:16px; text-align:center;">
    <div style="font-size:11px; color:#8a8aaa; text-transform:uppercase; letter-spacing:1px;">Tendance</div>
    <div style="font-size:22px; font-weight:700; margin:4px 0; color:{"#00c853" if current_price > ema_val else "#ff1744"};">{ema_signal}</div>
    <div style="font-size:11px; color:#8a8aaa;">EMA20: {ema_val:.5f}</div>
  </div>
  <div style="background:linear-gradient(135deg,#0d1117,#161b22); border:1px solid #333; border-radius:12px; padding:16px; text-align:center;">
    <div style="font-size:11px; color:#8a8aaa; text-transform:uppercase; letter-spacing:1px;">RSI7</div>
    <div style="font-size:22px; font-weight:700; margin:4px 0; color:{"#da70d6"};">{rsi_val:.1f}</div>
    <div style="font-size:11px; color:#8a8aaa;">{rsi_signal}</div>
  </div>
  <div style="background:linear-gradient(135deg,#0d1117,#161b22); border:1px solid #333; border-radius:12px; padding:16px; text-align:center;">
    <div style="font-size:11px; color:#8a8aaa; text-transform:uppercase; letter-spacing:1px;">Volatilité</div>
    <div style="font-size:22px; font-weight:700; margin:4px 0; color:#ffd700;">{atr_signal}</div>
    <div style="font-size:11px; color:#8a8aaa;">ATR: {atr_val:.5f}</div>
  </div>
</div>
"""

    # Prediction du moment
    engine = PredictionEngine()
    pred = engine.predict_2min(df)
    if pred:
        dir_icon = "🟢" if pred["direction"] == "PLUS_HAUT" else "🔴"
        score_color = "#00c853" if pred["score"] >= 0 else "#ff1744"
        output += f"""
<div style="background:linear-gradient(135deg,#0d1117,#1a2332); border:1px solid #333; border-radius:12px; padding:16px; margin:10px 0;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <span style="font-size:18px; font-weight:700;">🎯 Prédiction 2 min</span>
    <span style="font-size:20px; font-weight:700; color:{score_color};">{dir_icon} {pred['direction']}</span>
  </div>
  <div style="display:flex; gap:20px; margin-top:8px; font-size:13px; color:#8a8aaa;">
    <span>Score: <strong style="color:{score_color};">{pred['score']:+d}</strong></span>
    <span>Confiance: <strong style="color:#58a6ff;">{pred['confidence']}%</strong></span>
    <span>Session: <strong style="color:#e0e0e0;">{pred.get('session', '?')}</strong></span>
  </div>
</div>
"""
    else:
        output += "\n*Pas de prédiction pour l'instant.*\n"

    # Charts
    live_chart = generate_live_price_chart(df)

    # Session analysis from memory
    memory_path = _BOT_DIR / "memory.json"
    if memory_path.exists():
        try:
            with open(memory_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if history:
                # Build sessions stats on the fly
                sess_stats = {"Asian": {"ok": 0, "total": 0},
                             "London": {"ok": 0, "total": 0},
                             "New_York": {"ok": 0, "total": 0}}
                for e in history:
                    feat = e.get("features", {})
                    sess = feat.get("session", "?")
                    if sess in sess_stats:
                        sess_stats[sess]["total"] += 1
                        if e["outcome"]:
                            sess_stats[sess]["ok"] += 1
                stats_for_chart = {"sessions": sess_stats}
                session_chart = generate_session_chart(stats_for_chart)
            else:
                session_chart = None
        except Exception:
            session_chart = None
    else:
        session_chart = None

    return output, live_chart, session_chart


def backtest_ui(days: int) -> Tuple[str, Optional[Figure], Optional[Figure]]:
    """Backtest avec equity curve."""
    n_candles = min(days * 60 * 24, 2000)
    output = f"## 📊 Backtest EUR/USD — {days} jours\n\n"
    output += "Récupération des données yfinance...\n"

    df = fetch_historical_data(count=n_candles)
    if df.empty:
        return "❌ Pas de données.", None, None

    output += f"✅ {len(df)} bougies 1-minute chargées\n\n"
    predictions, stats = run_backtest(df)
    if not predictions:
        return "❌ Aucune prédiction générée.", None, None

    winrate = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
    total_pips = sum(p["change_pips"] for p in predictions if p["correct"])
    total_pips -= sum(abs(p["change_pips"]) for p in predictions if not p["correct"])
    avg_pips = sum(p["change_pips"] for p in predictions) / len(predictions) if predictions else 0

    output += "### 📈 Résultats Globaux\n\n"
    output += f"| Métrique | Valeur |\n|----------|-------|\n"
    output += f"| Total prédictions | **{stats['total']}** |\n"
    output += f"| ✅ Correctes | **{stats['correct']}** |\n"
    output += f"| ❌ Fausses | **{stats['wrong']}** |\n"
    output += f"| **Winrate** | **{winrate:.1f}%** |\n"
    output += f"| Score moyen | {sum(stats['scores'])/len(stats['scores']):.1f} |\n"
    output += f"| Confiance moyenne | {sum(stats['confidences'])/len(stats['confidences']):.1f}% |\n"
    output += f"| 💰 **P&L total (pips)** | **{total_pips:+.1f}** |\n"
    output += f"| 📊 Pips moyen par trade | {avg_pips:+.2f} |\n\n"

    output += "### 🔄 Par Direction\n\n"
    haut_wr = stats["haut_correct"] / stats["plus_haut"] * 100 if stats["plus_haut"] > 0 else 0
    bas_wr = stats["bas_correct"] / stats["plus_bas"] * 100 if stats["plus_bas"] > 0 else 0
    output += f"| Direction | Total | Correct | Winrate |\n"
    output += f"|-----------|-------|---------|--------|\n"
    output += f"| 🟢 PLUS HAUT | {stats['plus_haut']} | {stats['haut_correct']} | {haut_wr:.1f}% |\n"
    output += f"| 🔴 PLUS BAS | {stats['plus_bas']} | {stats['bas_correct']} | {bas_wr:.1f}% |\n\n"

    output += "### 🕐 Par Session\n\n"
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

    # Charts
    candle_chart = generate_candlestick_chart(df, predictions, max_candles=80)
    equity_chart = generate_equity_curve(predictions)

    return output, candle_chart, equity_chart


def live_analysis_ui() -> Tuple[str, Optional[Figure]]:
    """Analyse temps réel avec chart."""
    output = "## 🔍 Analyse Temps Réel EUR/USD\n\n"
    output += "*Données yfinance (15 min de retard)*\n\n"

    df = fetch_historical_data(count=200)
    if df.empty:
        return "❌ Pas de données.", None

    momentum = MomentumAnalyzer()
    liquidity = LiquidityAnalyzer()

    current_price = float(df["close"].iloc[-1])
    ema_val = momentum.ema(df)
    rsi_val = momentum.rsi(df)
    atr_val = momentum.atr(df)
    zones = liquidity.find_swing_highs_lows(df)

    output += f"### 💰 Prix Actuel\n\n**{current_price:.5f}** USD\n\n"
    output += "### 📊 Indicateurs Techniques\n\n"
    output += "| Indicateur | Valeur | Signal |\n|------------|-------|--------|\n"
    ema_signal = "🟢 HAUSSIER" if current_price > ema_val else "🔴 BAISSIER"
    output += f"| EMA20 | {ema_val:.5f} | {ema_signal} |\n"
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
    output += f"\n### 🏔️ Zones de Liquidité\n\n"
    output += f"| Plus haut (48) | Plus bas (48) |\n"
    output += f"| {zones['swing_high']:.5f} | {zones['swing_low']:.5f} |\n\n"

    engine = PredictionEngine()
    pred = engine.predict_2min(df)
    if pred:
        dir_icon = "🟢" if pred["direction"] == "PLUS_HAUT" else "🔴"
        score_color = "#00c853" if pred["score"] >= 0 else "#ff1744"
        output += f"### 🎯 Prédiction 2 minutes\n\n"
        output += f"**{dir_icon} {pred['direction']}** | "
        output += f"Score: <span style='color:{score_color};font-weight:bold;'>{pred['score']:+d}</span> | "
        output += f"Confiance: **{pred['confidence']}%**\n\n"
        output += "Détails:\n"
        for d in pred.get("details", []):
            output += f"- {d['icon']} {d['label']}: {d['value']}\n"
    else:
        output += "*Aucune prédiction pour l'instant*\n"

    chart = generate_live_price_chart(df)
    return output, chart


def memory_analysis_ui() -> Tuple[str, Optional[Figure], Optional[Figure], Optional[Figure]]:
    """Analyse mémoire avec graphiques."""
    history, stats = load_memory_data()
    if not history:
        return "❌ Aucune donnée mémoire trouvée.", None, None, None

    output = f"## 💾 Analyse de Performance — memory.json\n\n"
    output += f"**{stats['total']} entrées** | Winrate: **{stats['winrate']}%**\n\n"

    output += "### 📊 Synthèse\n\n"
    output += f"| Métrique | Valeur |\n|----------|-------|\n"
    output += f"| Total entrées | {stats['total']} |\n"
    output += f"| ✅ Correctes | {stats['correct']} |\n"
    output += f"| ❌ Fausses | {stats['wrong']} |\n"
    output += f"| **Winrate** | **{stats['winrate']}%** |\n\n"

    output += "### 🕐 Par Session\n\n"
    for sess in ["Asian", "London", "New_York"]:
        s = stats.get("sessions", {}).get(sess, {"ok": 0, "total": 0})
        if s["total"] > 0:
            wr = s["ok"] / s["total"] * 100
            bar = "█" * int(wr / 10) + "░" * (10 - int(wr / 10))
            output += f"| {sess} | {bar} {wr:.1f}% ({s['ok']}/{s['total']}) |\n"

    output += "\n### 🎯 Par Direction\n\n"
    for d in ["PLUS_HAUT", "PLUS_BAS"]:
        s = stats.get("directions", {}).get(d, {"ok": 0, "total": 0})
        if s["total"] > 0:
            wr = s["ok"] / s["total"] * 100
            label = "🟢 HAUT" if d == "PLUS_HAUT" else "🔴 BAS"
            output += f"| {label} | {wr:.1f}% ({s['ok']}/{s['total']}) |\n"

    output += "\n### 🔥 Top Profils (pattern|session)\n\n"
    profiles = stats.get("profiles", {})
    sorted_profiles = sorted(profiles.items(), key=lambda x: -x[1]["total"])[:10]
    for prof, s in sorted_profiles:
        if s["total"] > 0:
            wr = s["ok"] / s["total"] * 100
            bar = "█" * int(wr / 10) + "░" * (10 - int(wr / 10))
            output += f"| `{prof}` | {bar} {wr:.1f}% ({s['ok']}/{s['total']}) |\n"

    # Charts
    session_chart = generate_session_chart(stats)
    hourly_chart = generate_hourly_heatmap(stats)
    evolution_chart = generate_evolution_chart(history)

    return output, session_chart, hourly_chart, evolution_chart


def ml_training_ui() -> Tuple[str, Optional[Figure]]:
    """ML avec feature importance chart."""
    history, stats = load_memory_data()
    if not history:
        return "❌ Aucune donnée mémoire.", None

    output = f"## 🤖 Entraînement du Modèle ML\n\n"
    output += f"**{len(history)} échantillons disponibles**\n\n"

    if not HAS_SKLEARN:
        return "❌ scikit-learn n'est pas installé.", None

    output += "Lancement de l'entraînement RandomForest...\n\n"
    result = train_ml_on_memory(history)
    if result is None:
        return "❌ L'entraînement a échoué.", None

    if result["status"] == "not_enough_data":
        return f"⚠️ Pas assez de données ({result['samples']} échantillons). Minimum requis: 10.", None

    output += "### ✅ Modèle Entraîné avec Succès\n\n"
    output += f"**Échantillons:** {result['samples']}\n"
    output += f"**Algorithme:** RandomForest (100 arbres, profondeur max=5)\n\n"

    output += "### 🔮 Tests de Prédiction (ML Override)\n\n"
    output += "| # | Outcome réel | ML Override | Direction ML |\n"
    output += "|---|-------------|-------------|-------------|\n"
    for i, t in enumerate(result.get("test_results", [])[:10], 1):
        outcome_icon = "✅" if t["actual_outcome"] else "❌"
        override_icon = "🔄" if t["ml_override"] else "➖"
        output += f"| {i} | {outcome_icon} | {override_icon} | {t['ml_direction']} |\n"

    # Feature importance chart
    importances = result.get("feature_importances", [])
    feature_names = result.get("features_names", [])
    fi_chart = generate_feature_importance_chart(importances, feature_names)

    return output, fi_chart


def strategy_explainer_ui() -> str:
    return """## 📚 Explication de la Stratégie

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


### 📊 Dashboard Metrics Explained

| Métrique | Description |
|----------|-------------|
| **Winrate** | % de prédictions correctes sur le total |
| **P&L simulé** | P&L hypothétique en pips (1 pip = 1 unité) |
| **Sessions** | Asian (00-09h), London (08-17h), New_York (13-22h) UTC |
| **Overlap** | Chevauchement London × New_York (13-17h UTC) |
| **Score** | Force du signal basée sur les bougies + momentum + liquidité |
| **Confiance** | Estimation dérivée du score (50-95%) |
"""


def discord_preview_ui() -> str:
    return """## 💬 Aperçu Messages Discord

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
✅ **Verification — CORRECT**

Prediction: PLUS HAUT dans 2 min
Prix T+0: 1.10452
Prix T+2: 1.10500
📈 Variation: +4.8 pips

━━━━━━━━━━━━━━━━━━
```

### ❌ Vérification Échouée
```
❌ **Verification — FAUX**

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


# ====================================================================
# BOT CONTROL (lancer/arrêter le bot depuis le dashboard)
# ====================================================================

BOT_LOG_FILE = _BOT_DIR / "bot.log"

def start_bot_ui() -> Tuple[str, str]:
    """Démarrer le bot de trading en arrière-plan."""
    global _bot_process, _bot_start_time

    if _bot_process is not None:
        # Vérifier si le processus est toujours en vie
        if _bot_process.poll() is None:
            return "⚠️ Le bot est déjà en cours d'exécution !", get_bot_logs_raw()
        _bot_process = None

    try:
        _bot_process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(_BOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        _bot_start_time = time.time()

        # Attendre un peu pour vérifier que ça ne crash pas immédiatement
        time.sleep(1)
        if _bot_process.poll() is not None:
            out, _ = _bot_process.communicate(timeout=2)
            _bot_process = None
            return f"❌ Le bot a crashé au démarrage :\n```\n{out[-500:]}\n```", ""

        return "✅ **Bot démarré avec succès !**\n\nLe bot tourne maintenant en arrière-plan.\nConsulte les logs ci-dessous pour voir son activité.", get_bot_logs_raw()

    except Exception as e:
        _bot_process = None
        return f"❌ Erreur au démarrage : {e}", ""


def stop_bot_ui() -> Tuple[str, str]:
    """Arrêter le bot de trading."""
    global _bot_process, _bot_start_time

    if _bot_process is None:
        return "⚠️ Aucun bot en cours d'exécution.", get_bot_logs_raw()

    try:
        # Arrêt progressif
        if sys.platform == "win32":
            _bot_process.terminate()
        else:
            os.kill(_bot_process.pid, signal.SIGTERM)

        # Attendre jusqu'à 5 secondes
        try:
            _bot_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Forcer l'arrêt
            if sys.platform == "win32":
                _bot_process.kill()
            else:
                os.kill(_bot_process.pid, signal.SIGKILL)
            _bot_process.wait(timeout=2)

        runtime = time.time() - _bot_start_time if _bot_start_time else 0
        runtime_str = f"{int(runtime // 60)}m {int(runtime % 60)}s"
        _bot_process = None
        _bot_start_time = None

        return f"✅ **Bot arrêté.** (durée: {runtime_str})", get_bot_logs_raw()

    except Exception as e:
        _bot_process = None
        return f"❌ Erreur à l'arrêt : {e}", get_bot_logs_raw()


def get_bot_logs_raw() -> str:
    """Récupérer les dernières lignes du log."""
    if not BOT_LOG_FILE.exists():
        return "*Aucun log pour l'instant.*"
    try:
        with open(BOT_LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        # Dernières 80 lignes
        return "".join(lines[-80:])
    except Exception as e:
        return f"*Erreur lecture log: {e}*"


def get_bot_status_ui() -> Tuple[str, str]:
    """Obtenir le statut et les logs du bot."""
    global _bot_process, _bot_start_time

    is_running = _bot_process is not None and _bot_process.poll() is None
    logs = get_bot_logs_raw()

    if is_running:
        runtime = time.time() - _bot_start_time if _bot_start_time else 0
        runtime_str = f"{int(runtime // 60)}m {int(runtime % 60)}s"
        # Compter les prédictions dans memory.json
        memory_path = _BOT_DIR / "memory.json"
        pred_count = 0
        if memory_path.exists():
            try:
                with open(memory_path, "r") as f:
                    pred_count = len(json.load(f))
            except Exception:
                pass

        status = """
<div style="background:linear-gradient(135deg,#0d1117,#0a2a1a); border:1px solid #00c85344; border-radius:12px; padding:20px; margin:10px 0;">
  <div style="display:flex; align-items:center; gap:12px;">
    <div style="width:14px; height:14px; border-radius:50%; background:#00c853; box-shadow:0 0 10px #00c85388; animation:pulse 2s infinite;"></div>
    <span style="font-size:20px; font-weight:700; color:#00c853;">🟢 BOT EN COURS D'EXÉCUTION</span>
  </div>
  <div style="display:flex; gap:30px; margin-top:12px; font-size:14px; color:#8a8aaa;">
    <span>⏱️ **Durée:** {runtime_str}</span>
    <span>📊 **Prédictions:** {pred_count}</span>
    <span>🆔 **PID:** {_bot_process.pid}</span>
  </div>
</div>

<style>
@keyframes pulse {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.4; }}
}}
</style>
"""
        return status, logs
    else:
        _bot_process = None
        status = """
<div style="background:linear-gradient(135deg,#0d1117,#2a0a0a); border:1px solid #ff174444; border-radius:12px; padding:20px; margin:10px 0;">
  <div style="display:flex; align-items:center; gap:12px;">
    <div style="width:14px; height:14px; border-radius:50%; background:#4a4a6a;"></div>
    <span style="font-size:20px; font-weight:700; color:#ff1744;">🔴 BOT ARRÊTÉ</span>
  </div>
  <div style="margin-top:8px; font-size:13px; color:#8a8aaa;">
    Clique sur **▶️ Démarrer le Bot** pour lancer le trading automatique.
  </div>
</div>
"""
        return status, logs


def load_config_preview() -> str:
    """Afficher la configuration actuelle du bot."""
    try:
        import config
        return f"""
| Paramètre | Valeur |
|-----------|--------|
| Provider données | `{config.WS_DATA_PROVIDER}` |
| Symbole | `{config.SYMBOL}` |
| Seuil score | `{config.SCORE_THRESHOLD_OLD}` |
| Min ATR | `{config.MIN_ATR}` |
| ML activé | `{config.ML_ENABLED}` |
| Discord Webhook | `{config.DISCORD_WEBHOOK_URL[:30]}...` |
| Lookback bougies | `{config.LOOKBACK_CANDLES}` |
| Intervalle analyse | `{config.ANALYSIS_INTERVAL}s` |
"""
    except Exception as e:
        return f"❌ Erreur lecture config: {e}"


# ====================================================================
# CSS (amélioré)
# ====================================================================

CSS = """
<style>
    .container { max-width: 1300px; margin: auto; }
    .gradio-container { background: #0d1117 !important; }

    /* Cartes KPI */
    .kpi-card {
        background: linear-gradient(135deg, #0d1117, #161b22);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    /* Tables */
    table {
        width: 100%;
        border-collapse: collapse;
        background: #161b22;
        border-radius: 8px;
        overflow: hidden;
    }
    th, td {
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid #21262d;
        color: #e0e0e0;
    }
    th {
        background: #1c2333;
        font-weight: 600;
        color: #8a8aaa;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.5px;
    }
    tr:hover td {
        background: #1c2333;
    }
    code {
        background: #1c2333;
        padding: 2px 8px;
        border-radius: 4px;
        color: #58a6ff;
        font-size: 0.9em;
    }
    h1 { color: #e0e0e0; }
    h2, h3 { color: #e0e0e0; font-weight: 600; }
    p { color: #c0c0c0; }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-green { background: #00c85322; color: #00c853; border: 1px solid #00c85344; }
    .badge-red { background: #ff174422; color: #ff1744; border: 1px solid #ff174444; }
    .badge-blue { background: #58a6ff22; color: #58a6ff; border: 1px solid #58a6ff44; }

    /* Tabs */
    .tab-nav button {
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 10px 18px !important;
        border-radius: 8px 8px 0 0 !important;
    }

    /* Buttons */
    button {
        border-radius: 8px !important;
        transition: all 0.2s !important;
    }
    button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #484848;
        font-size: 12px;
    }
</style>
"""

# ====================================================================
# BUILD APP
# ====================================================================

def build_app():
    with gr.Blocks(
        title="Forex Algo Bot - Dashboard",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
        ),
        css=CSS,
        theme_mode="dark",
    ) as app:

        # Header
        gr.Markdown("""
        # 📈 **Forex Algo Bot** — EUR/USD M1

        <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px;">
            <span class="badge badge-green">🤖 Stratégie: Bougies + EMA20/RSI7</span>
            <span class="badge badge-blue">🧠 ML: RandomForest</span>
            <span class="badge badge-blue">📊 Data: yfinance</span>
        </div>
        """)

        with gr.Row():
            gr.Markdown(f"""
            <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1rem; align-items:center;">
                <span style="font-size:13px;">🤖 Bot: {'<span class="badge badge-green">OK</span>' if BOT_OK else '<span class="badge badge-red">N/A</span>'}</span>
                <span style="font-size:13px;">🧠 ML: {'<span class="badge badge-green">OK</span>' if HAS_SKLEARN else '<span class="badge badge-red">N/A</span>'}</span>
            </div>
            """)

        with gr.Tabs():
            # — Tab 0: Overview —
            with gr.TabItem("🏠 Vue d'ensemble", id=0):
                with gr.Row():
                    overview_btn = gr.Button("🔄 Actualiser", variant="primary", size="lg", scale=2)
                with gr.Row():
                    overview_kpis = gr.Markdown("Cliquez pour charger...")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        overview_chart = gr.Plot(label="Graphique Prix + RSI")
                    with gr.Column(scale=1):
                        overview_sessions = gr.Plot(label="Performance par Session")
                overview_btn.click(
                    fn=overview_ui,
                    inputs=[],
                    outputs=[overview_kpis, overview_chart, overview_sessions],
                )

            # — Tab 1: Live Analysis (amélioré) —
            with gr.TabItem("🔍 Analyse Temps Réel", id=1):
                with gr.Row():
                    live_btn = gr.Button("🔄 Actualiser l'Analyse", variant="primary", size="lg", scale=2)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        live_output = gr.Markdown("Cliquez pour analyser...")
                    with gr.Column(scale=3):
                        live_chart = gr.Plot(label="Prix + RSI Chart")
                live_btn.click(
                    fn=live_analysis_ui,
                    inputs=[],
                    outputs=[live_output, live_chart],
                )

            # — Tab 2: Backtest (amélioré) —
            with gr.TabItem("📊 Backtest", id=2):
                with gr.Row():
                    days_slider = gr.Slider(
                        minimum=1, maximum=5, value=2, step=1,
                        label="Nombre de jours à backtester",
                    )
                with gr.Row():
                    backtest_btn = gr.Button("🚀 Lancer le Backtest", variant="primary", size="lg")
                with gr.Row():
                    backtest_output = gr.Markdown("Configurez et lancez...")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        backtest_chart = gr.Plot(label="Bougies + Prédictions")
                    with gr.Column(scale=1):
                        equity_chart = gr.Plot(label="Simulation P&L")
                backtest_btn.click(
                    fn=backtest_ui,
                    inputs=[days_slider],
                    outputs=[backtest_output, backtest_chart, equity_chart],
                )

            # — Tab 3: Memory Analysis (amélioré) —
            with gr.TabItem("💾 Analyse Mémoire", id=3):
                with gr.Row():
                    memory_btn = gr.Button("📊 Analyser memory.json", variant="primary", size="lg")
                with gr.Row():
                    memory_output = gr.Markdown("Analysez les performances...")
                with gr.Row(equal_height=True):
                    memory_session_chart = gr.Plot(label="Winrate par Session")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        memory_hourly_chart = gr.Plot(label="Winrate par Heure")
                    with gr.Column(scale=1):
                        memory_evolution_chart = gr.Plot(label="Évolution du Winrate")
                memory_btn.click(
                    fn=memory_analysis_ui,
                    inputs=[],
                    outputs=[memory_output, memory_session_chart, memory_hourly_chart, memory_evolution_chart],
                )

            # — Tab 4: ML Training (amélioré) —
            with gr.TabItem("🤖 Apprentissage ML", id=4):
                with gr.Row():
                    ml_btn = gr.Button("🧠 Entraîner le Modèle", variant="primary", size="lg")
                with gr.Row():
                    ml_output = gr.Markdown("Cliquez pour entraîner...")
                with gr.Row():
                    ml_fi_chart = gr.Plot(label="Importance des Features")
                ml_btn.click(
                    fn=ml_training_ui,
                    inputs=[],
                    outputs=[ml_output, ml_fi_chart],
                )

            # — Tab 5: Strategy —
            with gr.TabItem("📚 Stratégie", id=5):
                strategy_output = gr.Markdown(strategy_explainer_ui())

            # — Tab 6: Bot Control (NOUVEAU) —
            with gr.TabItem("🤖 Contrôle du Bot", id=6):
                with gr.Row():
                    gr.Markdown("## 🤖 Lancer / Arrêter le Bot de Trading\n\nDémarre le bot en arrière-plan. Les logs s'affichent en direct.")

                with gr.Row():
                    config_md = gr.Markdown(load_config_preview())

                with gr.Row(equal_height=True):
                    start_btn = gr.Button("▶️ Démarrer le Bot", variant="primary", size="lg", scale=2)
                    stop_btn = gr.Button("⏹️ Arrêter le Bot", variant="stop", size="lg", scale=1)
                    refresh_btn = gr.Button("🔄 Rafraîchir", size="lg", scale=1)

                with gr.Row():
                    bot_status = gr.Markdown("""
<div style="background:linear-gradient(135deg,#0d1117,#2a0a0a); border:1px solid #ff174444; border-radius:12px; padding:20px; margin:10px 0;">
  <div style="display:flex; align-items:center; gap:12px;">
    <div style="width:14px; height:14px; border-radius:50%; background:#4a4a6a;"></div>
    <span style="font-size:20px; font-weight:700; color:#ff1744;">🔴 BOT ARRÊTÉ</span>
  </div>
  <div style="margin-top:8px; font-size:13px; color:#8a8aaa;">
    Clique sur **▶️ Démarrer le Bot** pour lancer le trading automatique.
  </div>
</div>
""")

                with gr.Row():
                    bot_logs = gr.Markdown("*Les logs apparaîtront ici...*")

                # Boutons → actions
                start_btn.click(
                    fn=start_bot_ui,
                    inputs=[],
                    outputs=[bot_status, bot_logs],
                )
                stop_btn.click(
                    fn=stop_bot_ui,
                    inputs=[],
                    outputs=[bot_status, bot_logs],
                )
                refresh_btn.click(
                    fn=get_bot_status_ui,
                    inputs=[],
                    outputs=[bot_status, bot_logs],
                )

            # — Tab 7: Discord —
            with gr.TabItem("💬 Discord", id=7):
                discord_output = gr.Markdown(discord_preview_ui())

        # Footer
        gr.Markdown("""
        ---
        <div class="footer">
            <p>
                📈 <strong>Forex Algo Bot</strong> — EUR/USD M1 |
                Stratégie: Bougies + EMA20/RSI7 + ML RandomForest |
                Propulsé par yfinance, scikit-learn & Gradio
            </p>
        </div>
        """)

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("  FOREX ALGO BOT — Dashboard de Performance")
    print("=" * 60)
    print(f"\n  Bot modules: {'OK' if BOT_OK else 'NON DISPONIBLE'}")
    print(f"  scikit-learn: {'OK' if HAS_SKLEARN else 'NON DISPONIBLE'}")
    print(f"  yfinance: disponible")
    print()

    app = build_app()
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)
