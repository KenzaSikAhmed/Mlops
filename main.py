import numpy as np
import pandas as pd
import joblib
import argparse
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Set your MLflow tracking URI
mlflow.set_experiment("Churn Prediction with SVM")


from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--data_path", type=str, default="merged_churn1.csv.csv", help="Chemin du jeu de données")
    args = parser.parse_args()

    if args.prepare:
        print("📊 Préparation des données en cours...")
        X_train, X_test, y_train, y_test = prepare_data(args.data_path)  # ✅ FIXED unpacking issue
        print("✅ Données préparées avec succès.")
        print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]} échantillons")
        print(f"Taille de l'ensemble de test : {X_test.shape[0]} échantillons")

    if args.train:
        print("🚀 Entraînement du modèle en cours...")
        X_train, X_test, y_train, y_test = prepare_data(args.data_path)  # ✅ FIXED
        
        # Démarrer une expérience MLflow
        with mlflow.start_run():
            # Entraînement du modèle
            model = train_model(X_train, y_train)

            # Enregistrer les hyperparamètres utilisés
            mlflow.log_param("Model Type", "AdaBoostClassifier")  # Remplacer par le type de modèle réel
            mlflow.log_param("Num_estimators", 50)  # Exemple d'hyperparamètre (changer selon votre modèle)
            
            # Enregistrer les métriques
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

            # Sauvegarder le modèle avec MLflow
            mlflow.sklearn.log_model(model, "model")

            print("✅ Modèle entraîné et sauvegardé avec MLflow.")
            print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    if args.evaluate:
        print("🔍 Évaluation du modèle en cours...")
        X_train, X_test, y_train, y_test = prepare_data(args.data_path)  # ✅ FIXED
        model = load_model()
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()

