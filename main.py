import numpy as np
import pandas as pd
import joblib
import argparse
import logging
import os
from sklearn.metrics import accuracy_score

# Vérifier si on est dans GitHub Actions
IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

if not IS_GITHUB_ACTIONS:
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Remplace par ton URI MLflow si besoin
    mlflow.set_experiment("Churn Prediction with SVM")
else:
    print("⚠️ MLflow désactivé sur GitHub Actions.")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Importation des fonctions du fichier model_pipeline.py
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle SVM")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle SVM")
    parser.add_argument(
        "--data_path",
        type=str,
        default="merged_churn1.csv.csv",
        help="Chemin du jeu de données",
    )
    args = parser.parse_args()

    if args.prepare:
        try:
            print("📊 Préparation des données en cours...")
            X_train, X_test, y_train, y_test = prepare_data(args.data_path)
            print("✅ Données préparées avec succès.")
            print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]} échantillons")
            print(f"Taille de l'ensemble de test : {X_test.shape[0]} échantillons")
        except Exception as e:
            print(f"❌ Erreur lors de la préparation des données : {e}")
            return

    if args.train:
        try:
            print("🚀 Entraînement du modèle SVM en cours...")
            X_train, X_test, y_train, y_test = prepare_data(args.data_path)

            if not IS_GITHUB_ACTIONS:
                # Lancement de l'expérience MLflow
                with mlflow.start_run():
                    # Entraînement du modèle SVM avec GridSearchCV
                    model = train_model(X_train, y_train)

                    # Enregistrer les hyperparamètres optimaux
                    mlflow.log_param("Model Type", "SVM")
                    mlflow.log_param("Best C", model.best_params_["C"])
                    mlflow.log_param("Best Kernel", model.best_params_["kernel"])

                    # Évaluation avant sauvegarde
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    mlflow.log_metric("accuracy", accuracy)

                    # Sauvegarde du modèle
                    save_model(model, "model.pkl")
                    mlflow.sklearn.log_model(model, "model")

                    print(f"✅ Modèle SVM entraîné et sauvegardé avec MLflow. Accuracy : {accuracy:.4f}")
                    print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
            else:
                print("⚠️ MLflow désactivé sur GitHub Actions. Entraînement du modèle sans tracking.")
                model = train_model(X_train, y_train)
                save_model(model, "model.pkl")

        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement du modèle SVM : {e}")  

    if args.evaluate:
        try:
            print("🔍 Évaluation du modèle SVM en cours...")
            if not os.path.exists("model.pkl"):
                raise FileNotFoundError("❌ Le fichier model.pkl est introuvable. Entraînez le modèle d'abord.")

            X_train, X_test, y_train, y_test = prepare_data(args.data_path)
            model = load_model("model.pkl")
            accuracy = evaluate_model(model, X_test, y_test)
            print(f"✅ Évaluation terminée. Accuracy : {accuracy:.4f}")

        except Exception as e:
            print(f"❌ Erreur lors de l'évaluation : {e}")

if __name__ == "__main__":
    main()

