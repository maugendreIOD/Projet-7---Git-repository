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

# Copier les fichiers requirements.txt depuis le dossier Docker et installer les dépendances
COPY Docker/requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

# Créer la structure du package Python
RUN mkdir -p /app/app
RUN touch /app/app/__init__.py

# Copier le code de l'application
COPY app/main.py /app/app/
COPY app/linear_regression_model.pkl /app/app/
COPY app/preprocessing_pipeline.pkl /app/app/
COPY tests/test_api.py /app/tests/
COPY tests/test_data.json /app/tests/

# Add a simple pytest.ini file
RUN echo "[pytest]\npythonpath = /app" > pytest.ini

# Make sure our app directory is in PYTHONPATH
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Exposer le port sur lequel Flask écoute (8080 pour Google Cloud Run)
EXPOSE 8080

# Lancer l'application Flask
CMD ["python", "app/main.py"]
