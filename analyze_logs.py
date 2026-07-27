#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse complete de tous les fichiers de log du Forex Algo Bot"""

import sys
import re
from pathlib import Path
from collections import defaultdict, Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path("G:/forex-algo-bot")

# ─── Patterns ───────────────────────────────────────────────────────────────
RE_PREDICTION = re.compile(
    r"Pred:\s*(PLUS_HAUT|PLUS_BAS)\s*\|\s*Score:\s*(-?\d+)"
)
RE_VERIFICATION = re.compile(
    r"Verif:\s*(OK|KO)\s*\(([+-]?\d+\.?\d*)\s*pips\)"
)
RE_RETRAIN = re.compile(r"Retraining du modele ML")
RE_TRAIN = re.compile(r"Modele ML entraine:")
RE_TRAIN_INFO = re.compile(
    r"Modele ML entraine:\s*(\d+)\s*echantillons,\s*accuracy=(\d+)%"
)
RE_ML_OVERRIDE = re.compile(r"ML override:\s*(\S+)\s*->\s*(\S+)")
RE_MEMORY_COUNT = re.compile(r"Memoire:\s*(\d+) antecedents")
RE_DUPLICATE = re.compile(r"Duplicate ignore")
RE_ERROR = re.compile(r"ERROR")
RE_WARNING = re.compile(r"WARNING")
RE_HISTORIC = re.compile(r"Chargement historique")
RE_CONNECT = re.compile(r"Connexion")
RE_SENT = re.compile(r"Prediction envoyee")
RE_PROVIDER = re.compile(r"Provider:\s*(\S+)")

# Format de base: "2026-07-07 10:25:22,123 [INFO] logger: message"
# Mais certains logs ont des formats differents. On va parser les messages.


def analyze_file(path: Path, label: str):
    if not path.exists():
        print(f"  [MANQUANT] {label}")
        return None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    total = len(lines)
    if total == 0:
        print(f"  [VIDE] {label}")
        return None

    # Compteurs
    predictions = []
    verifications = {"OK": 0, "KO": 0, "total": 0}
    ml_retrains = 0
    ml_trains = 0
    ml_accuracies = []
    ml_overrides = 0
    duplicates = 0
    errors = 0
    warnings = 0
    predicted_sent = 0
    memory_count = 0

    # Pour l'evolution temporelle
    dates = []
    pred_dates = []

    for line in lines:
        # Prediction
        m = RE_PREDICTION.search(line)
        if m:
            predictions.append({
                "direction": m.group(1),
                "score": int(m.group(2)),
                "line": line.strip(),
            })
            # extraire la date
            d = line[:19] if len(line) > 19 else ""
            pred_dates.append(d)
            dates.append(d)

        # Verification
        m = RE_VERIFICATION.search(line)
        if m:
            ok = m.group(1) == "OK"
            pips = float(m.group(2))
            verifications["total"] += 1
            if ok:
                verifications["OK"] += 1
            else:
                verifications["KO"] += 1

        # ML events
        if RE_RETRAIN.search(line):
            ml_retrains += 1
        if RE_TRAIN.search(line):
            ml_trains += 1
        m = RE_TRAIN_INFO.search(line)
        if m:
            ml_accuracies.append(int(m.group(2)))

        # ML override
        if RE_ML_OVERRIDE.search(line):
            ml_overrides += 1

        # Duplicate
        if RE_DUPLICATE.search(line):
            duplicates += 1

        # Errors / Warnings
        if RE_ERROR.search(line):
            errors += 1
        if RE_WARNING.search(line):
            warnings += 1

        # Prediction sent
        if RE_SENT.search(line):
            predicted_sent += 1

        # Memory
        m = RE_MEMORY_COUNT.search(line)
        if m:
            memory_count = int(m.group(1))

        # Dates
        d = line[:19] if len(line) > 19 else ""
        if d and d[4] == "-":
            dates.append(d)

    # Calcul des stats
    total_preds = len(predictions)
    winrate = 0
    if verifications["total"] > 0:
        winrate = verifications["OK"] / verifications["total"] * 100

    # Score moyen
    scores = [p["score"] for p in predictions]
    avg_score = sum(scores) / len(scores) if scores else 0

    # Direction counts
    dirs = Counter(p["direction"] for p in predictions)

    # Periode
    first_date = min(dates) if dates else "?"
    last_date = max(dates) if dates else "?"

    # Provider
    provider = ""
    for line in lines:
        m = RE_PROVIDER.search(line)
        if m:
            provider = m.group(1)
            break

    return {
        "label": label,
        "path": path,
        "total_lines": total,
        "first_date": first_date,
        "last_date": last_date,
        "provider": provider,
        "total_predictions": total_preds,
        "predicted_sent": predicted_sent,
        "verifications": verifications,
        "winrate_pct": round(winrate, 1),
        "avg_score": round(avg_score, 1),
        "direction_breakdown": {"PLUS_HAUT": dirs.get("PLUS_HAUT", 0), "PLUS_BAS": dirs.get("PLUS_BAS", 0)},
        "duplicates": duplicates,
        "errors": errors,
        "warnings": warnings,
        "ml_trains": ml_trains,
        "ml_retrains": ml_retrains,
        "ml_accuracies": ml_accuracies,
        "ml_overrides": ml_overrides,
        "memory_count": memory_count,
    }


def print_header(title: str):
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def print_stats(stats: dict):
    if stats is None:
        return

    print(f"\n  Fichier     : {stats['path'].name}")
    print(f"  Periode     : {stats['first_date']}  ->  {stats['last_date']}")
    print(f"  Provider    : {stats['provider'] or '?'}")
    print(f"  Lignes      : {stats['total_lines']:,}")
    print()

    v = stats["verifications"]
    print(f"  Predictions : {stats['total_predictions']} (dont {stats['predicted_sent']} envoyees a Discord)")
    print(f"  Verifications: {v['total']}  ({v['OK']} OK / {v['KO']} KO)")
    print(f"  Winrate     : {stats['winrate_pct']:.1f}%")
    print(f"  Score moyen : {stats['avg_score']}")

    d = stats["direction_breakdown"]
    total_d = d["PLUS_HAUT"] + d["PLUS_BAS"]
    if total_d > 0:
        pct_haut = d["PLUS_HAUT"] / total_d * 100
        pct_bas = d["PLUS_BAS"] / total_d * 100
        print(f"  Directions  : HAUT {d['PLUS_HAUT']} ({pct_haut:.0f}%)  |  BAS {d['PLUS_BAS']} ({pct_bas:.0f}%)")
    print(f"  Duplicates  : {stats['duplicates']}")
    print(f"  Erreurs     : {stats['errors']}")
    print(f"  Warnings    : {stats['warnings']}")

    if stats["ml_trains"] > 0 or stats["ml_retrains"] > 0:
        print(f"  ML Trains   : {stats['ml_trains']} (retrains: {stats['ml_retrains']})")
        acc = stats["ml_accuracies"]
        if acc:
            print(f"  ML Accuracy : {sum(acc)/len(acc):.0f}% moyenne (derniere: {acc[-1]}%)")
        print(f"  ML Overrides: {stats['ml_overrides']}")


def print_comparison(all_stats):
    """Affiche un tableau de comparaison entre les differents logs"""
    valid = [s for s in all_stats if s is not None]
    if len(valid) < 2:
        return

    print()
    print("=" * 68)
    print("  TABLEAU COMPARATIF")
    print("=" * 68)
    print(f"  {'Fichier':<25} {'Preds':>6} {'Verifs':>7} {'OK':>4} {'WR':>6} {'Score':>6} {'Err':>5} {'ML':>4}")
    print("  " + "-" * 68)
    for s in sorted(valid, key=lambda x: x["total_predictions"], reverse=True):
        v = s["verifications"]
        wr = f"{s['winrate_pct']:.0f}%" if v["total"] > 0 else "-"
        ml = s["ml_trains"] + s["ml_retrains"]
        err = s["errors"] + s["warnings"]
        print(f"  {s['label']:<25} {s['total_predictions']:>6} {v['total']:>7} {v['OK']:>4} {wr:>6} {s['avg_score']:>6} {err:>5} {ml:>4}")


def top_predictions(lines: list, n: int = 10):
    """Affiche les N dernieres predictions avec verifications"""
    preds = []
    for line in lines:
        m = RE_PREDICTION.search(line)
        if m:
            preds.append({
                "text": line.strip(),
                "direction": m.group(1),
                "score": int(m.group(2)),
            })
    return preds[-n:]


# ─── Main ───────────────────────────────────────────────────────────────────
LOGS = [
    ("bot.log", "bot.log"),
    ("bot_dukascopy_30min.log", "dukascopy_30min"),
    ("bot_output.log", "bot_output"),
    ("bot_output_30min.log", "bot_output_30min"),
    ("bot_output_test.log", "bot_output_test"),
    ("bot_yfinance_final.log", "yfinance_final"),
    ("bot_yfinance_test.log", "yfinance_test"),
    ("bot_yfinance_v2.log", "yfinance_v2"),
    ("bot_yfinance_v3.log", "yfinance_v3"),
    ("bot_yfinance_v4.log", "yfinance_v4"),
    ("bot_yfinance_v5.log", "yfinance_v5"),
]

if __name__ == "__main__":
    all_stats = []

    for fname, label in LOGS:
        path = BASE / fname
        stats = analyze_file(path, label)
        all_stats.append(stats)

    # Afficher chaque log
    for stats in all_stats:
        print_header(f"Analyse: {stats['label'] if stats else '?'}")
        print_stats(stats)

    print_comparison(all_stats)

    # Afficher les dernieres lignes du bot.log
    main_log = BASE / "bot.log"
    if main_log.exists():
        print_header("Dernieres predictions (bot.log)")
        with open(main_log, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        last_preds = top_predictions(lines, 15)
        for i, p in enumerate(last_preds, 1):
            print(f"  {i:>2}. {p['text'][:90]}")

    print()
    print("=" * 68)
    print("  FIN DE L'ANALYSE")
    print("=" * 68)
