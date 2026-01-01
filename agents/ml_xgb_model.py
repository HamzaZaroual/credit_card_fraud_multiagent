# Copyright (c) 2026 Hamza Zaroual
# Licensed under the Apache License, Version 2.0
# See LICENSE file for details.


import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, precision_recall_curve, roc_auc_score,
    roc_curve, confusion_matrix, auc, f1_score, precision_score, recall_score, make_scorer
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ------------------- CLASSE XGB ML -------------------

class FraudDetectionML:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = None

    def load_and_prepare_data(self, filepath):
        """Charge et prépare les données"""
        data = pd.read_csv(filepath)
        if 'Class' not in data.columns:
            raise ValueError("La colonne 'Class' est absente du dataset")
        X = data.drop('Class', axis=1)
        y = data['Class']
        self.scaler = StandardScaler()
        X[['Time', 'Amount']] = self.scaler.fit_transform(X[['Time', 'Amount']])
        return X, y

    def split_data(self, X, y, test_size=0.3):
        """Sépare les données en train/val/test"""
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=self.random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def tune_model_with_smote(self, X_train, y_train):
        """Entraîne et optimise le modèle XGBoost avec SMOTE et RandomizedSearchCV"""
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=self.random_state)),
            ('xgb', XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=-1,
                tree_method='hist'
            ))
        ])

        param_distributions = {
            'smote__sampling_strategy': uniform(0.05, 0.15),
            'xgb__n_estimators': randint(100, 250),
            'xgb__max_depth': randint(3, 7),
            'xgb__learning_rate': uniform(0.03, 0.12),
            'xgb__subsample': uniform(0.7, 0.3),
            'xgb__colsample_bytree': uniform(0.7, 0.3),
            'xgb__min_child_weight': randint(1, 6),
            'xgb__gamma': uniform(0.0, 0.3)
        }

        scoring = {
            'F1_fraud': make_scorer(f1_score, pos_label=1),
            'Precision_fraud': make_scorer(precision_score, pos_label=1),
            'Recall_fraud': make_scorer(recall_score, pos_label=1),
            'ROC_AUC': 'roc_auc',
            'PR_AUC': make_scorer(self._auprc_score, needs_proba=True)
        }

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=30,
            scoring=scoring,
            refit='F1_fraud',
            cv=3,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        return self.model

    def _auprc_score(self, y_true, y_proba, **kwargs):
        """Calcule l'aire sous la courbe Precision-Recall"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        return auc(recall, precision)

    def evaluate(self, X, y):
        """Évalue le modèle sur un ensemble de données"""
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        precision, recall, _ = precision_recall_curve(y, y_proba)
        return {
            'y_true': y,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'roc_auc': roc_auc_score(y, y_proba),
            'pr_auc': auc(recall, precision),
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall
        }

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    def save_model(self, model_path):
        """Sauvegarde le modèle et le scaler"""
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"Modèle sauvegardé: {model_path}")
        if self.scaler:
            scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler sauvegardé: {scaler_path}")

    @classmethod
    def load_model(cls, model_path):
        """Charge un modèle sauvegardé et renvoie une instance de la classe"""
        loaded = cls()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier {model_path} n'existe pas")
        loaded.model = joblib.load(model_path)
        # Charger le scaler si disponible
        scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
        if os.path.exists(scaler_path):
            loaded.scaler = joblib.load(scaler_path)
        print(f"Modèle XGBoost chargé depuis {model_path}")
        return loaded


# ------------------- MAIN -------------------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '../data/processed/creditcard_processed.csv')
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    ml = FraudDetectionML()
    X, y = ml.load_and_prepare_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = ml.split_data(X, y)

    print("Entraînement et tuning du modèle XGBoost...")
    ml.tune_model_with_smote(X_train, y_train)

    print("Évaluation du modèle...")
    metrics = ml.evaluate(X_test, y_test)
    print(classification_report(y_test, metrics['y_pred'], digits=4))
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")

    ml.plot_confusion_matrix(metrics['y_true'], metrics['y_pred'])

    model_path = os.path.join(model_dir, 'xgb_fraud_detector.pkl')
    ml.save_model(model_path)

if __name__ == "__main__":
    main()
