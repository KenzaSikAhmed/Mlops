# Utiliser une image de base avec Python 3.12
FROM python:3.12

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers du projet dans l'image
COPY . /app

# Installer les dépendances du projet
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port 8000 pour FastAPI
EXPOSE 8000

# Lancer l'API avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
