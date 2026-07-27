---
title: Forex Algo Bot
emoji: 📈
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.31.0
app_file: app.py
pinned: false
---

# 📈 Forex Algo Bot — EUR/USD M1

Backtest, visualisation et analyse de performance du bot de trading Forex.

> **🚀 Déploiement gratuit disponible sur :**
> - **[Render.com](https://render.com)** (tier gratuit, lien ci-dessous)
> - **[Hugging Face Spaces](https://huggingface.co/spaces)** (abonnement PRO requis depuis 2025)

## 🔧 Fonctionnalités

| Feature | Description |
|---------|-------------|
| 🔍 **Analyse Temps Réel** | Analyse le marché EUR/USD avec yfinance (EMA20, RSI7, ATR, liquidité) |
| 📊 **Backtest** | Teste la stratégie sur les N derniers jours de données historiques |
| 💾 **Analyse Mémoire** | Analyse les performances depuis `memory.json` (winrate par profil/session) |
| 🤖 **Apprentissage ML** | Entraîne le RandomForest et visualise les overrides |
| 📚 **Stratégie** | Explication détaillée de la double stratégie |
| 💬 **Aperçu Discord** | Preview des messages envoyés sur Discord |

## 🧠 Stratégie

### Double confirmation
1. **Stratégie 1** — Comptage bougies consécutives + momentum + liquidité
2. **Stratégie 2** — Confirmation EMA20 + RSI7 + filtre ATR

Les 2 stratégies doivent être **d'accord** pour émettre un signal.

### Machine Learning (RandomForest)
Le modèle apprend de `memory.json` et peut **override** la direction si sa confiance ≥ 60%.

## 🚀 Déploiement

### Option 1 : Render.com (gratuit)

1. Va sur [github.com](https://github.com) et crée un dépôt
2. Pousse tout le dossier `forex-algo-bot/` sur GitHub
3. Va sur [dashboard.render.com](https://dashboard.render.com) → **New +** → **Blueprint**
4. Connecte ton dépôt GitHub
5. Render détecte automatiquement `render.yaml` et configure tout
6. ✅ Ton app sera en ligne sur `https://forex-algo-bot.onrender.com` en quelques minutes

> **Note :** Le tier gratuit de Render met l'app en veille après 15 min d'inactivité.
> Elle se réveille automatiquement au premier accès (5-10 secondes).

### Option 2 : Hugging Face Spaces (PRO requis)

```bash
# Copier les fichiers spécifiques HF dans le Space :
# - app.py
# - requirements-hf.txt → renommer en requirements.txt
# - README-hf.md → renommer en README.md
# - memory.json
# - strategy/ complet
# - learning/ complet
# - utils/ complet
# - notifier/ complet
```

## 📁 Structure des fichiers

```
forex-algo-bot/
├── app.py                 # Interface Gradio (point d'entrée)
├── requirements-hf.txt    # Dépendances cloud (Render/HF)
├── render.yaml            # Blueprint Render.com
├── README-hf.md           # Documentation
├── .gitignore-hf          # Gitignore
├── strategy/              # Moteur de prédiction
├── learning/              # ML + mémoire
├── utils/                 # Helpers
├── notifier/              # Discord webhook
├── data/                  # Providers de données
└── memory.json            # Données de performance
```

## ⚠️ Limitations sur le cloud

- Pas de connexion temps réel (MT5, OANDA, Dukascopy)
- Données yfinance avec ~15 min de retard
- Pas d'envoi Discord (le webhook fonctionne en local)
- Le ML est réentraîné à chaque redémarrage

---

*Propulsé par yfinance, scikit-learn & Gradio*
