#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse les performances du bot a partir de memory.json"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def load_memory(path="memory.json"):
    p = Path(path)
    if not p.exists():
        print(f"[FICHIER] {path} introuvable")
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze(history):
    if not history:
        print("[VIDE] Aucune donnee dans memory.json")
        return

    total = len(history)
    correct = sum(1 for e in history if e["outcome"])
    wrong = total - correct
    winrate_global = correct / total * 100 if total else 0

    print("=" * 60)
    print("STATISTIQUES DE PERFORMANCE -- Forex Algo Bot")
    print("=" * 60)
    print(f"\nWinrate global : {winrate_global:.1f}% ({correct}/{total})")
    print(f"  Correctes    : {correct}")
    print(f"  Fausses      : {wrong}")
    print()

    # ---- Par profil (pattern|session) ----
    print("-" * 60)
    print("PAR PATTERN (green_red | session)")
    print("-" * 60)
    profiles = defaultdict(lambda: {"ok": 0, "total": 0})
    for e in history:
        prof = e["profile"]
        profiles[prof]["total"] += 1
        if e["outcome"]:
            profiles[prof]["ok"] += 1

    for prof, stats in sorted(profiles.items()):
        wr = stats["ok"] / stats["total"] * 100
        bar = "#" * int(wr // 10) + "." * (10 - int(wr // 10))
        print(f"  {prof:<20} {bar} {wr:5.1f}% ({stats['ok']}/{stats['total']})")

    # ---- Par session ----
    print()
    print("-" * 60)
    print("PAR SESSION")
    print("-" * 60)
    sessions = defaultdict(lambda: {"ok": 0, "total": 0})
    for e in history:
        sess = e["features"].get("session", "?")
        sessions[sess]["total"] += 1
        if e["outcome"]:
            sessions[sess]["ok"] += 1

    for sess in ["Asian", "London", "New_York"]:
        stats = sessions.get(sess, {"ok": 0, "total": 0})
        if stats["total"] > 0:
            wr = stats["ok"] / stats["total"] * 100
            bar = "#" * int(wr // 10) + "." * (10 - int(wr // 10))
            print(f"  {sess:<15} {bar} {wr:5.1f}% ({stats['ok']}/{stats['total']})")
        else:
            print(f"  {sess:<15} --- aucun echantillon")

    # Overlap vs non-overlap
    overlap = {"ok": 0, "total": 0}
    no_overlap = {"ok": 0, "total": 0}
    for e in history:
        if e["features"].get("overlap"):
            overlap["total"] += 1
            if e["outcome"]:
                overlap["ok"] += 1
        else:
            no_overlap["total"] += 1
            if e["outcome"]:
                no_overlap["ok"] += 1
    if overlap["total"] > 0:
        wr = overlap["ok"] / overlap["total"] * 100
        print(f"\n  Overlap de sessions   : {wr:.1f}% ({overlap['ok']}/{overlap['total']})")
    if no_overlap["total"] > 0:
        wr = no_overlap["ok"] / no_overlap["total"] * 100
        print(f"  Sans overlap          : {wr:.1f}% ({no_overlap['ok']}/{no_overlap['total']})")

    # ---- Par direction ----
    print()
    print("-" * 60)
    print("PAR DIRECTION PREDITE")
    print("-" * 60)
    directions = defaultdict(lambda: {"ok": 0, "total": 0})
    for e in history:
        d = e["features"].get("direction", "?")
        directions[d]["total"] += 1
        if e["outcome"]:
            directions[d]["ok"] += 1

    for d in ["PLUS_HAUT", "PLUS_BAS"]:
        stats = directions.get(d, {"ok": 0, "total": 0})
        if stats["total"] > 0:
            wr = stats["ok"] / stats["total"] * 100
            label = "HAUT" if d == "PLUS_HAUT" else "BAS"
            bar = "#" * int(wr // 10) + "." * (10 - int(wr // 10))
            print(f"  {label:<10} {bar} {wr:5.1f}% ({stats['ok']}/{stats['total']})")

    # ---- Stats ATR ----
    atr_vals = [e["features"].get("atr", 0) for e in history]
    if atr_vals:
        print(f"\nATR moyen : {sum(atr_vals)/len(atr_vals):.5f}")
        print(f"  ATR min  : {min(atr_vals):.5f}")
        print(f"  ATR max  : {max(atr_vals):.5f}")

    # ---- Evolution dans le temps ----
    print()
    print("-" * 60)
    print("EVOLUTION (par blocs de 20 predictions)")
    print("-" * 60)
    block_size = 20
    for i in range(0, total, block_size):
        block = history[i:i + block_size]
        ok = sum(1 for e in block if e["outcome"])
        wr = ok / len(block) * 100
        bar = "#" * int(wr // 10) + "." * (10 - int(wr // 10))
        print(f"  #{i//block_size + 1:<3} [{i:>4}-{i+len(block)-1:>4}] {bar} {wr:5.1f}% ({ok}/{len(block)})")

    # ---- Score moyen ----
    scores = [e["features"].get("score", 0) for e in history]
    if scores:
        ok_scores = [s for s, e in zip(scores, history) if e["outcome"]]
        ko_scores = [s for s, e in zip(scores, history) if not e["outcome"]]
        if ok_scores:
            print(f"\nScore moyen (correct) : {sum(ok_scores)/len(ok_scores):.1f}")
        if ko_scores:
            print(f"  Score moyen (faux)    : {sum(ko_scores)/len(ko_scores):.1f}")
        print(f"  Score moyen (global)  : {sum(scores)/len(scores):.1f}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    history = load_memory()
    analyze(history)
