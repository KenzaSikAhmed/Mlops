import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

MODEL_FILENAME = "model.pkl"
SCALER_FILENAME = "scaler.pkl"
FEATURES_FILENAME = "model_features.pkl"

SELECTED_FEATURES = [
    "Account length", "Area code", "Customer service calls", "International plan",
    "Number vmail messages", "Total day calls", "Total day charge", "Total day minutes",
    "Total night calls", "Total night charge", "Total night minutes",
    "Total eve calls", "Total eve charge", "Total eve minutes",
    "Total intl calls", "Voice mail plan"
]

def prepare_data(data_path='merged_churn1.csv.csv'):
    """ Load, preprocess, and split data into train/test sets with ONLY 16 features. """
    data = pd.read_csv(data_path)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['International plan'] = label_encoder.fit_transform(data['International plan'])
    data['Voice mail plan'] = label_encoder.fit_transform(data['Voice mail plan'])
    data['Churn'] = label_encoder.fit_transform(data['Churn'])
    
    X = data[SELECTED_FEATURES]
    y = data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    joblib.dump(scaler, SCALER_FILENAME)
    joblib.dump(SELECTED_FEATURES, FEATURES_FILENAME)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("Début de l'entraînement du modèle SVM...")
    
    svm_model = SVC()
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=cv, scoring='accuracy')
    
    grid_search.fit(X_train, y_train)
    
    print("Meilleurs paramètres trouvés :", grid_search.best_params_)
    print("Meilleur score d'exactitude :", grid_search.best_score_)
    
    return grid_search

def evaluate_model(model, X_test, y_test):
    """ Evaluate the trained model on the test set. """
    y_pred = model.predict(X_test)
    
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f" Accuracy: {accuracy:.4f}")
    
    return accuracy

def save_model(model, filename=MODEL_FILENAME):
    """ Save the trained model as a .pkl file. """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename=MODEL_FILENAME):
    """ Load a saved model. """
    return joblib.load(filename)

