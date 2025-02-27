from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from model_pipeline import prepare_data, train_model, save_model
import os  # Ajout de l'importation de la bibliothèque os

# Chargement du modèle, du scaler et des features
MODEL_FILENAME = "model.pkl"
SCALER_FILENAME = "scaler.pkl"
FEATURES_FILENAME = "model_features.pkl"

try:
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    SELECTED_FEATURES = joblib.load(FEATURES_FILENAME)
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=f"Erreur de chargement des fichiers : {e}")

# Initialisation de l'API FastAPI
app = FastAPI()

# Modèle de requête pour la prédiction
class PredictionInput(BaseModel):
    features: list  # Liste des features en entrée

# Modèle de requête pour le réentraînement
class RetrainInput(BaseModel):
    data_path: str = "merged_churn.csv"  
    hyperparameters: dict  # Hyperparamètres pour le réentraînement

# Encodage des variables catégorielles
label_encoder = LabelEncoder()

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Vérification du nombre de features
        if len(data.features) != len(SELECTED_FEATURES):
            raise HTTPException(status_code=400, detail=f"Nombre incorrect de features. Attendu : {len(SELECTED_FEATURES)}, Reçu : {len(data.features)}")

        # Création d'un DataFrame avec les features
        input_data = pd.DataFrame([data.features], columns=SELECTED_FEATURES)

        # Vérification et encodage des variables catégorielles
        if 'International plan' in input_data.columns and 'Voice mail plan' in input_data.columns:
            input_data['International plan'] = label_encoder.fit_transform(input_data['International plan'])
            input_data['Voice mail plan'] = label_encoder.fit_transform(input_data['Voice mail plan'])
        else:
            raise HTTPException(status_code=400, detail="Les variables catégorielles attendues 'International plan' ou 'Voice mail plan' sont manquantes.")

        # Normalisation des features
        scaled_features = scaler.transform(input_data)

        # Prédiction
        prediction = model.predict(scaled_features)
        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {e}")
        
@app.post("/retrain")
def retrain(data: RetrainInput):
    try:
        # Vérification du chemin absolu du fichier de données
        absolute_data_path = os.path.abspath(data.data_path)
        if not os.path.exists(absolute_data_path):
            raise HTTPException(status_code=400, detail="Le fichier de données spécifié est introuvable.")

        # Préparation des données
        X_train, X_test, y_train, y_test, _ = prepare_data(data.data_path)

        # Entraînement du modèle avec les nouveaux hyperparamètres
        model = train_model(X_train, y_train, **data.hyperparameters)

        # Sauvegarde du modèle réentraîné
        save_model(model)
        return {"message": "Le modèle a été réentraîné avec succès."}

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Le fichier de données est introuvable.")
    except Exception as e:
        print(f"Erreur interne : {e}")  # Afficher l'erreur dans la console
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {e}")
