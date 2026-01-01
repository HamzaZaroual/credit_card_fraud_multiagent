# Copyright (c) 2026 Hamza Zaroual
# Licensed under the Apache License, Version 2.0
# See LICENSE file for details.



import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dl_model import FraudDetectionDL
from ml_xgb_model import FraudDetectionML
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def final_pipeline(data_path, xgb_model_path, dl_model_path, output_path):
    # ------------------- 1️1- Chargement des données -------------------
    data = pd.read_csv(data_path)
    X = data.drop('Class', axis=1)
    
    # ------------------- 2️2- Préprocessing -------------------
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    
    # ------------------- 3️3- Chargement des modèles -------------------
    xgb_model = FraudDetectionML.load_model(xgb_model_path)
    dl_model = FraudDetectionDL.load_model(dl_model_path, input_shape=X.shape[1])
    
    # ------------------- 4️4- Prédictions -------------------
    xgb_proba = xgb_model.model.predict_proba(X)[:, 1]  # probabilité de fraude xgb
    dl_proba = dl_model.predict_proba(X).flatten()      # probabilité de fraude dl
    
    # ------------------- 5️5- Post-processing -------------------
    xgb_pred = (xgb_proba > 0.5).astype(int)
    dl_pred = (dl_proba > dl_model.best_threshold).astype(int)
    
    # Moyenne des probabilités pour la prédiction finale
    final_proba = (xgb_proba + dl_proba) / 2
    final_pred = (final_proba > 0.5).astype(int)
    
    # ------------------- 6️6- Création du DataFrame de sortie -------------------
    result = pd.DataFrame({
        'Time': data['Time'],
        'Amount': data['Amount'],
        'Class': data['Class'],
        'fraud_ml': xgb_proba,
        'fraud_dl': dl_proba,
        'fraud_final_proba': final_proba,
        'fraud_final_label': final_pred
    })
    
    # ------------------- 7- Sauvegarde -------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Résultats sauvegardés dans : {output_path}")
    print(result.head())
    
    # ------------------- 8- Matrice de confusion -------------------
    cm = confusion_matrix(data['Class'], final_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion - Pipeline final")
    plt.show()
    
    # ------------------- 9- Statistiques de détection -------------------
    tn, fp, fn, tp = cm.ravel()
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Nombre de fraudes détectées (final) : {final_pred.sum()} / {data['Class'].sum()} vraies fraudes")
    
    return result


# ------------------- UTILISATION -------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '../data/processed/creditcard_processed.csv')
    xgb_model_path = os.path.join(base_dir, './models/xgb_fraud_detector.pkl')
    dl_model_path = os.path.join(base_dir, './models/fraud_detection_dl_model.h5')
    output_path = os.path.join(base_dir, '../results/final_predictions.csv')
    
    final_pipeline(data_path, xgb_model_path, dl_model_path, output_path)
