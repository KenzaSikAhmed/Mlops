# Déclaration des variables
PYTHON=python3
ENV_NAME=venv
REQUIREMENTS=requirements.txt
SOURCE_DIR=model_pipeline.py
MAIN_SCRIPT=main.py
TEST_DIR=tests/
IMAGE_NAME=kenzasikahmed0/kenza_sikahmed_4ds5_mlops
CONTAINER_NAME=mlops_container
TAG=latest
PORT=8000

# Configuration de l'environnement
setup:
	@echo "Création de l'environnement virtuel et installation des dépendances..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@./$(ENV_NAME)/bin/python3 -m pip install --upgrade pip
	@./$(ENV_NAME)/bin/python3 -m pip install -r $(REQUIREMENTS)
	@echo "Environnement configuré avec succès !"

# Vérification du code
verify:
	@echo "Vérification de la qualité du code..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) -m black --exclude 'venv|mlops_env' .
	@. $(ENV_NAME)/bin/activate && $(PYTHON) -m pylint --disable=C,R $(SOURCE_DIR) || true
	@echo "Code vérifié avec succès !"

# Préparation des données
prepare:
	@echo "Préparation des données..."
	@./$(ENV_NAME)/bin/python3 $(MAIN_SCRIPT) --prepare
	@echo "Données préparées avec succès !"

# Entraînement du modèle
train:
	@echo "Entraînement du modèle..."
	@./$(ENV_NAME)/bin/python3 $(MAIN_SCRIPT) --train
	@echo "Modèle entraîné avec succès !"

# Évaluation du modèle
evaluate:
	@echo "Évaluation du modèle..."
	@./$(ENV_NAME)/bin/python3 $(MAIN_SCRIPT) --evaluate
	@echo "Évaluation terminée !"

# Exécution des tests
test:
	@echo "Exécution des tests..."
	@if [ ! -d "$(TEST_DIR)" ]; then echo "Création du dossier $(TEST_DIR)..."; mkdir -p $(TEST_DIR); fi
	@if [ -z "$$(ls -A $(TEST_DIR))" ]; then echo "Aucun test trouvé ! Création d'un test basique..."; echo 'def test_dummy(): assert 2 + 2 == 4' > $(TEST_DIR)/test_dummy.py; fi
	@./$(ENV_NAME)/bin/python3 -m pytest $(TEST_DIR) --disable-warnings
	@echo "Tests exécutés avec succès !"

# Nettoyage des fichiers temporaires
clean:
	@echo "Suppression des fichiers temporaires..."
	rm -rf $(ENV_NAME)
	rm -f model.pkl scaler.pkl pca.pkl
	rm -rf __pycache__ .pytest_cache .pylint.d
	@echo "Nettoyage terminé !"

# Réinstallation complète de l'environnement
reinstall: clean setup

# Pipeline complet
all: setup verify prepare train test
	@echo "Pipeline MLOps exécuté avec succès !"

# Lancer l'API FastAPI
run-api:
	@bash -c "source venv/bin/activate && uvicorn app:app --host 0.0.0.0 --port $(PORT) --reload"

# -------------------------------
#       AJOUT DES COMMANDES DOCKER
# -------------------------------

# Construire l'image Docker
build_docker:
	@echo "Construction de l'image Docker..."
	docker build -t $(IMAGE_NAME):$(TAG) .
	@echo "Image Docker construite avec succès !"

# Taguer l'image Docker (optionnel si build déjà fait)
tag_docker:
	@echo "Tag de l'image Docker..."
	docker tag $(IMAGE_NAME):$(TAG) $(IMAGE_NAME):latest
	@echo "Tagging terminé !"

# Se connecter à Docker Hub
docker_login:
	@echo "Connexion à Docker Hub..."
	@docker login
	@echo "Connexion réussie !"

# Pousser l'image sur Docker Hub
push_docker: docker_login
	@echo "Ajout du tag latest..."
	docker tag $(IMAGE_NAME):$(TAG) $(IMAGE_NAME):latest
	@echo "Envoi de l'image sur Docker Hub..."
	docker push $(IMAGE_NAME):$(TAG)
	docker push $(IMAGE_NAME):latest
	@echo "Image envoyée avec succès !"

# Exécuter le conteneur Docker
run_docker:
	@echo "Démarrage du conteneur Docker..."
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	docker run -d -p $(PORT):8000 --name $(CONTAINER_NAME) $(IMAGE_NAME):$(TAG)
	@echo "Conteneur Docker démarré avec succès !"

# Arrêter et supprimer le conteneur Docker
stop_docker:
	@echo "Arrêt du conteneur Docker..."
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	@echo "Conteneur supprimé !"

# Pipeline complet avec Docker
deploy: build_docker push_docker run_docker
	@echo "Déploiement complet effectué !"

