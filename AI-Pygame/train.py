"""
train.py — Trainingsscript voor het neurale netwerk.
Leest training_data.csv, traint een model, slaat op als model.keras
Voert ook experimenten uit en toont grafieken.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

CSV_FILE    = "training_data.csv"
MODEL_FILE  = "model.keras"

# ── Data laden ────────────────────────────────────────────────────────────────

def load_data(csv_file=CSV_FILE):
    df = pd.read_csv(csv_file)
    print(f"Geladen: {len(df)} rijen  |  Klasse-verdeling:\n{df['action'].value_counts()}\n")
    X = df[["player_x", "block_x", "block_y", "delta_x", "delta_y"]].values
    y = df["action"].values
    return X, y


def prepare_labels(y):
    lb = LabelBinarizer()
    y_enc = lb.fit_transform(y)
    # LabelBinarizer geeft bij 2 klassen een 1D array; zorg altijd voor 3 kolommen
    if y_enc.shape[1] < 3:
        full = np.zeros((len(y), 3))
        for i, cls in enumerate(lb.classes_):
            full[:, cls] = y_enc[:, i] if y_enc.shape[1] > 1 else y_enc[:, 0]
        y_enc = full
    return y_enc


# ── Model bouwen ──────────────────────────────────────────────────────────────

def build_model(hidden1=32, hidden2=16, input_dim=5):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden1, activation="relu"),
        layers.Dense(hidden2, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_evaluate(X, y_enc, epochs=80, label="model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )

    model = build_model()
    print(f"── Training: {label} (input_dim={X.shape[1]}) ──")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    train_acc = history.history["accuracy"][-1]
    val_acc   = history.history["val_accuracy"][-1]
    print(f"\n  Train accuracy : {train_acc:.4f}")
    print(f"  Test  accuracy : {val_acc:.4f}\n")

    return model, history, train_acc, val_acc


def plot_history(history, label="model"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Trainingsverloop — {label}", fontsize=13)

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validatie")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Validatie")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"plot_{label.replace(' ', '_')}.png", dpi=120)
    plt.show()
    print(f"  Grafiek opgeslagen: plot_{label.replace(' ', '_')}.png")


# ── Experimenten ──────────────────────────────────────────────────────────────

def experiment_dataset_size(X, y_enc):
    """Experiment 2: varieer de dataset grootte."""
    sizes  = [200, 500, len(X)]
    results = []
    for n in sizes:
        n = min(n, len(X))
        Xi, yi = X[:n], y_enc[:n]
        _, hist, tr, te = train_and_evaluate(Xi, yi, epochs=60, label=f"n={n}")
        results.append((n, tr, te))

    # Tabel
    print("\n─── Experiment: Dataset grootte ───")
    print(f"{'Samples':>10}  {'Train acc':>10}  {'Test acc':>10}")
    for n, tr, te in results:
        print(f"{n:>10}  {tr:>10.4f}  {te:>10.4f}")

    # Grafiek
    ns   = [r[0] for r in results]
    trs  = [r[1] for r in results]
    tes  = [r[2] for r in results]
    plt.figure(figsize=(7, 4))
    plt.plot(ns, trs, "o-", label="Train")
    plt.plot(ns, tes, "s-", label="Test")
    plt.title("Accuracy vs. Dataset grootte")
    plt.xlabel("Aantal trainingssamples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_dataset_size.png", dpi=120)
    plt.show()


def experiment_epochs(X, y_enc):
    """Experiment 3: varieer het aantal epochs."""
    epoch_list = [20, 50, 100]
    results = []
    for ep in epoch_list:
        _, hist, tr, te = train_and_evaluate(X, y_enc, epochs=ep, label=f"epochs={ep}")
        results.append((ep, tr, te))

    print("\n─── Experiment: Epochs ───")
    print(f"{'Epochs':>8}  {'Train acc':>10}  {'Test acc':>10}")
    for ep, tr, te in results:
        print(f"{ep:>8}  {tr:>10.4f}  {te:>10.4f}")

    eps  = [r[0] for r in results]
    trs  = [r[1] for r in results]
    tes  = [r[2] for r in results]
    plt.figure(figsize=(7, 4))
    plt.plot(eps, trs, "o-", label="Train")
    plt.plot(eps, tes, "s-", label="Test")
    plt.title("Accuracy vs. Aantal epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_epochs.png", dpi=120)
    plt.show()


# ── Hoofd ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CSV_FILE):
        print(f"Fout: {CSV_FILE} niet gevonden. Speel eerst het spel (game.py).")
        return

    X, y_raw = load_data()
    y_enc    = prepare_labels(y_raw)

    # Hoofd-training (alle data, 80 epochs)
    model, history, _, _ = train_and_evaluate(X, y_enc, epochs=80, label="hoofdmodel")
    plot_history(history, label="hoofdmodel")

    # Model opslaan
    model.save(MODEL_FILE)
    print(f"\nModel opgeslagen als: {MODEL_FILE}")

    # Experimenten
    print("\n\n=== EXPERIMENTEN ===\n")
    experiment_dataset_size(X, y_enc)
    experiment_epochs(X, y_enc)

    print("\nKlaar! Voer nu ai_play.py uit om de AI het spel te laten spelen.")


if __name__ == "__main__":
    main()
