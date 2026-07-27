# ===================================================================
# Dockerfile — Forex Algo Bot Dashboard (Gradio)
# Déploiement sur Render.com via Docker
# ===================================================================
# Construction:
#   docker build -t forex-dashboard .
#   docker run -p 7860:7860 forex-dashboard
# ===================================================================

FROM python:3.12-slim

# Évite l'écriture de fichiers .pyc et bufferisation de stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Installer les dépendances système minimales (matplotlib a besoin de libs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copier et installer les dépendances Python
COPY requirements-hf.txt .
RUN pip install --no-cache-dir -r requirements-hf.txt

# Copier tout le projet
COPY . .

# S'assurer que le dossier __pycache__ n'est pas créé par le host
RUN find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Le port utilisé par Gradio (sera détecté automatiquement par Render)
EXPOSE 7860

# Commande de démarrage
# app.py lit PORT de l'environnement (Render le définit automatiquement)
# Fallback sur 7860 si PORT n'est pas défini
CMD ["python", "app.py"]
