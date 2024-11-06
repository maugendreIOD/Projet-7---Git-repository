# Utiliser une image Python 3.10 slim
FROM python:3.10-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Créer un répertoire pour l'application
WORKDIR /app

# Copier les fichiers requirements.txt et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code dans le conteneur
COPY . /app

# Exposer le port sur lequel Flask écoute (8080 pour Google Cloud Run)
EXPOSE 8080

# Lancer l'application Flask
CMD ["python", "main.py"]

