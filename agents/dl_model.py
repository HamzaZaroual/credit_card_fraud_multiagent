# Copyright (c) 2026 Hamza Zaroual
# Licensed under the Apache License, Version 2.0
# See LICENSE file for details.


import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    auc,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tempfile

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ============================ DEVICE ============================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        device = "GPU"
    except RuntimeError as e:
        print(e)
        device = "CPU"
else:
    device = "CPU"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print(f"Using device: {device}")

# ============================ FOCAL LOSS ============================
def binary_focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return tf.reduce_mean(-alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
    loss.__name__ = "binary_focal_loss"
    return loss

# ============================ MODEL CLASS ============================
class FraudDetectionDL:

    def __init__(self, input_shape, learning_rate=1e-3):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.scaler = None
        self.history = None
        self.best_threshold = 0.5

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.input_shape,)),

            Dense(128, activation="relu", kernel_regularizer=l2(1e-3)),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation="relu", kernel_regularizer=l2(1e-3)),
            BatchNormalization(),
            Dropout(0.3),

            Dense(32, activation="relu", kernel_regularizer=l2(1e-3)),
            BatchNormalization(),
            Dropout(0.2),

            Dense(16, activation="relu", kernel_regularizer=l2(1e-3)),
            Dropout(0.2),

            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=Adam(self.learning_rate),
            loss=binary_focal_loss(),
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall")
            ]
        )
        return model

    # ======================== LOAD MODEL ========================
    @classmethod
    def load_model(cls, model_path, input_shape):
        instance = cls(input_shape=input_shape)
        instance.model = tf.keras.models.load_model(
            model_path,
            custom_objects={"binary_focal_loss": binary_focal_loss()}
        )
        scaler_path = os.path.join(os.path.dirname(model_path), "scaler.pkl")
        if os.path.exists(scaler_path):
            instance.scaler = joblib.load(scaler_path)
        threshold_path = os.path.join(os.path.dirname(model_path), "best_threshold.txt")
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as f:
                instance.best_threshold = float(f.read())
        return instance

    # ======================== TRAIN ========================
    def train(self, X_train, y_train, X_val, y_val,
              epochs=20, batch_size=512, model_path=None):

        callbacks = [
            EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        if model_path:
            callbacks.append(
                ModelCheckpoint(
                    model_path,
                    monitor="val_pr_auc",
                    mode="max",
                    save_best_only=True,
                    verbose=1
                )
            )

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

    # ======================== FINE TUNING (TEMP ONLY) ========================
    def fine_tune(self, X_train, y_train, X_val, y_val,
                  max_trials=10, epochs=10, batch_size=512):

        temp_dir = tempfile.mkdtemp()  # dossier temporaire pour Keras-Tuner

        def build_model(hp):
            l2_reg = hp.Choice("l2", [1e-4, 5e-4, 1e-3])
            lr = hp.Choice("lr", [1e-4, 3e-4, 1e-3])

            model = Sequential([
                Input(shape=(self.input_shape,)),

                Dense(hp.Choice("u1", [64, 128, 256]),
                      activation="relu", kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(hp.Choice("d1", [0.2, 0.3, 0.4])),

                Dense(hp.Choice("u2", [32, 64, 128]),
                      activation="relu", kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(hp.Choice("d2", [0.2, 0.3])),

                Dense(hp.Choice("u3", [16, 32, 64]),
                      activation="relu", kernel_regularizer=l2(l2_reg)),
                Dropout(hp.Choice("d3", [0.1, 0.2])),

                Dense(1, activation="sigmoid")
            ])

            model.compile(
                optimizer=Adam(lr),
                loss=binary_focal_loss(),
                metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")]
            )
            return model

        tuner = kt.RandomSearch(
            build_model,
            objective=kt.Objective("val_pr_auc", direction="max"),
            max_trials=max_trials,
            directory=temp_dir,
            project_name="tmp_tuning",
            overwrite=True
        )

        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=8,
                restore_best_weights=True
            )],
            verbose=0
        )

        best_hp = tuner.get_best_hyperparameters(1)[0]
        self.model = tuner.hypermodel.build(best_hp)
        self.learning_rate = best_hp.get("lr")

        # Nettoyer le dossier temporaire
        shutil.rmtree(temp_dir)

    # ======================== EVALUATION ========================
    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)

    def predict(self, X, threshold=None):
        t = self.best_threshold if threshold is None else threshold
        return (self.predict_proba(X) > t).astype(int)

    def evaluate(self, X, y, threshold):
        y_pred = self.predict_proba(X)
        y_bin = (y_pred > threshold).astype(int)

        precision, recall, _ = precision_recall_curve(y, y_pred)

        return {
            "classification_report": classification_report(
                y, y_bin, digits=4,
                target_names=["Non-Fraud", "Fraud"]
            ),
            "confusion_matrix": confusion_matrix(y, y_bin),
            "roc_auc": roc_auc_score(y, y_pred),
            "pr_auc": auc(recall, precision),
            "f1_score": f1_score(y, y_bin)
        }

    def find_optimal_threshold(self, X, y):
        y_pred = self.predict_proba(X)
        thresholds = np.linspace(0.1, 0.9, 100)
        f1s = [f1_score(y, (y_pred > t).astype(int)) for t in thresholds]
        self.best_threshold = thresholds[np.argmax(f1s)]
        return self.best_threshold

    # ======================== PLOTS ========================
    def plot_training_history(self):
        h = self.history.history
        plt.plot(h["loss"], label="train")
        plt.plot(h["val_loss"], label="val")
        plt.title("Loss")
        plt.legend()
        plt.show()

        plt.plot(h["pr_auc"], label="train PR-AUC")
        plt.plot(h["val_pr_auc"], label="val PR-AUC")
        plt.title("PR-AUC")
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, X, y):
        cm = confusion_matrix(y, self.predict(X))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=["Non-Fraud", "Fraud"],
                    yticklabels=["Non-Fraud", "Fraud"])
        plt.show()

    def plot_roc_curve(self, X, y):
        y_pred = self.predict_proba(X)
        fpr, tpr, _ = roc_curve(y, y_pred)
        auc_val = roc_auc_score(y, y_pred)
        plt.plot(fpr, tpr, label=f"AUC={auc_val:.4f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.legend()
        plt.show()

    # ======================== SAVE MODEL ========================
    def save_model(self, model_path, threshold_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        if self.scaler:
            joblib.dump(self.scaler, os.path.join(os.path.dirname(model_path), "scaler.pkl"))
        with open(threshold_path, "w") as f:
            f.write(str(self.best_threshold))


# ============================ MAIN ============================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../data/processed/creditcard_processed.csv")
    model_path = "/home/utilisateur/credit_card_fraud_multiagent/agents/models/fraud_detection_dl_model.h5"
    threshold_path = "/home/utilisateur/credit_card_fraud_multiagent/agents/models/best_threshold.txt"

    data = pd.read_csv(data_path)

    X = data.drop("Class", axis=1)
    y = data["Class"]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )

    scaler = StandardScaler()
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_val[["Time", "Amount"]] = scaler.transform(X_val[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    model = FraudDetectionDL(input_shape=X_train.shape[1])
    model.scaler = scaler

    # Fine-tuning et entrainement final
    model.fine_tune(X_train, y_train, X_val, y_val)
    model.train(X_train, y_train, X_val, y_val)

    # Trouver le meilleur seuil et évaluer
    threshold = model.find_optimal_threshold(X_val, y_val)
    metrics = model.evaluate(X_test, y_test, threshold)

    print(metrics["classification_report"])
    print("ROC AUC:", metrics["roc_auc"])
    print("PR AUC :", metrics["pr_auc"])
    print("F1     :", metrics["f1_score"])

    model.plot_training_history()
    model.plot_confusion_matrix(X_test, y_test)
    model.plot_roc_curve(X_test, y_test)

    # Sauvegarde finale du meilleur modèle et seuil
    model.save_model(model_path, threshold_path)

if __name__ == "__main__":
    main()
