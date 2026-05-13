
from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Variabili per gestione dell'import dinamico di scikit-learn
_SKLEARN_IMPORT_ERROR: Optional[BaseException] = None

if TYPE_CHECKING:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    _SKLEARN_AVAILABLE = True
else:
    # Classificatore KNN da scikit-learn — KNN trattato a lezione (lec istance-based)
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            classification_report, confusion_matrix
        )
        _SKLEARN_AVAILABLE = True
    except ImportError as exc:
        KNeighborsClassifier: Any = None
        train_test_split: Any = None
        accuracy_score: Any = None
        precision_score: Any = None
        recall_score: Any = None
        f1_score: Any = None
        classification_report: Any = None
        confusion_matrix: Any = None
        _SKLEARN_AVAILABLE = False
        _SKLEARN_IMPORT_ERROR = exc


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "Scikit-learn non è installato. "
            "Installa il pacchetto con: pip install scikit-learn"
        ) from _SKLEARN_IMPORT_ERROR

# -----------------------------------------------------------------------
# Configurazione
# -----------------------------------------------------------------------

CELL_SIZE     = 28        # Dimensione di ogni cella in pixel (come MNIST)
SAMPLES_CLASS = 200       # Numero di campioni per classe nel dataset
DATASET_DIR   = Path("dataset")   # Cartella del dataset generato
MODEL_PATH    = Path("knn_model.pkl")  # Percorso dove salvare il modello
K_NEIGHBORS   = 3         # Valore di k per KNN

# Simboli riconosciuti dal classificatore e loro etichetta di classe
SYMBOLS = {
    "S": "START",
    "D": "DELIVERY",
    "W": "WIND",
    "X": "BLOCKED",
    "E": "EMPTY",
}


# -----------------------------------------------------------------------
# 1. Generazione del Dataset
# -----------------------------------------------------------------------

def _render_symbol(symbol: str, size: int = CELL_SIZE) -> np.ndarray:
    # Sfondo bianco, testo nero (come MNIST)
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)

    # Font di dimensione casuale per variabilità
    font_size = random.randint(14, 20)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        try:
            # Windows path
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Posizione con leggero jitter casuale
    bbox  = draw.textbbox((0, 0), symbol, font=font)
    tw    = bbox[2] - bbox[0]
    th    = bbox[3] - bbox[1]
    x     = (size - tw) // 2 + random.randint(-2, 2)
    y     = (size - th) // 2 + random.randint(-2, 2)
    draw.text((x, y), symbol, fill=0, font=font)

    # Rotazione leggera (-10 a +10 gradi) per robustezza
    angle = random.uniform(-10, 10)
    img   = img.rotate(angle, fillcolor=255)

    # Rumore gaussiano leggero per simulare imperfezioni
    arr  = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 8, arr.shape)
    arr  = np.clip(arr + noise, 0, 255)

    # Normalizzazione [0, 1] — standard per classificatori su immagini
    arr = arr / 255.0

    # Appiattimento: ogni immagine 28x28 diventa un vettore di 784 feature
    # Questo è il vettore di attributi x del training set (lec classificazione)
    return arr.flatten()


def generate_dataset(
    dataset_dir: Path = DATASET_DIR,
    samples_per_class: int = SAMPLES_CLASS,
    cell_size: int = CELL_SIZE,
) -> None:
    """
    Genera il dataset di training: per ogni simbolo crea
    'samples_per_class' immagini con perturbazioni casuali.

    Struttura risultante:
      dataset/S/  -> 200 immagini .png del simbolo S
      dataset/D/  -> 200 immagini .png del simbolo D
      ...
    """
    print(f"[ML] Generazione dataset in '{dataset_dir}' ...")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for symbol in SYMBOLS:
        class_dir = dataset_dir / symbol
        class_dir.mkdir(exist_ok=True)

        for i in range(samples_per_class):
            arr = _render_symbol(symbol, cell_size)
            # Ricostruisce immagine 28x28 per salvarla su disco
            img_arr = (arr.reshape(cell_size, cell_size) * 255).astype(np.uint8)
            img = Image.fromarray(img_arr, mode="L")
            img.save(class_dir / f"{symbol}_{i:04d}.png")

        print(f"  {symbol} ({SYMBOLS[symbol]}): {samples_per_class} campioni generati")

    print(f"[ML] Dataset generato: {len(SYMBOLS)} classi x {samples_per_class} = "
          f"{len(SYMBOLS) * samples_per_class} immagini totali")


# -----------------------------------------------------------------------
# 2. Caricamento Dataset
# -----------------------------------------------------------------------

def load_dataset(dataset_dir: Path = DATASET_DIR) -> tuple[np.ndarray, np.ndarray]:
    """
    Carica le immagini dal dataset e le restituisce come:
      X: matrice (n_samples, 784) — vettori di feature (pixel normalizzati)
      y: array (n_samples,)       — etichette di classe (S, D, W, X, E)

    Il formato (X, y) corrisponde al training set del search problem di
    classificazione: ogni coppia (x_i, y_i) e' un esempio etichettato.
    """
    X, y = [], []

    for symbol in SYMBOLS:
        class_dir = dataset_dir / symbol
        if not class_dir.exists():
            continue
        for img_path in class_dir.glob("*.png"):
            img = Image.open(img_path).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            X.append(arr.flatten())
            y.append(symbol)

    return np.array(X, dtype=np.float32), np.array(y)


# -----------------------------------------------------------------------
# 3. Training e Valutazione del KNN
# -----------------------------------------------------------------------

def train_knn(
    dataset_dir: Path = DATASET_DIR,
    k: int = K_NEIGHBORS,
    test_size: float = 0.2,
    model_path: Path = MODEL_PATH,
) -> KNeighborsClassifier:
    
    print(f"\n[ML] Caricamento dataset da '{dataset_dir}' ...")
    X, y = load_dataset(dataset_dir)
    print(f"[ML] Dataset: {len(X)} campioni, {len(np.unique(y))} classi")
    _require_sklearn()

    # Train/test split (80% training, 20% test) — lec valutazione modello
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[ML] Split: {len(X_train)} training, {len(X_test)} test")

    # Addestramento KNN — lazy learner: memorizza tutto il training set
    print(f"\n[ML] Addestramento KNN (k={k}, metrica=euclidea) ...")
    _require_sklearn()
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)

    # Valutazione sul test set
    y_pred = knn.predict(X_test)

    print("\n[ML] ===== RISULTATI CLASSIFICAZIONE =====")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  F-measure: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

    print("\n[ML] Report per classe:")
    print(classification_report(y_test, y_pred, target_names=list(SYMBOLS.keys()), zero_division=0))

    print("[ML] Matrice di confusione:")
    print(confusion_matrix(y_test, y_pred, labels=list(SYMBOLS.keys())))

    # Salva il modello addestrato su disco per riutilizzarlo in image_parser
    with open(model_path, "wb") as f:
        pickle.dump(knn, f)
    print(f"\n[ML] Modello salvato in '{model_path}'")

    return knn


# -----------------------------------------------------------------------
# 4. Predizione su singola cella
# -----------------------------------------------------------------------

def load_model(model_path: Path = MODEL_PATH) -> KNeighborsClassifier:
    """Carica il modello KNN salvato su disco."""
    _require_sklearn()
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Modello non trovato in '{model_path}'. "
            f"Esegui prima train_knn() o ml_classifier.py direttamente."
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_cell(
    cell_img: np.ndarray,
    knn: KNeighborsClassifier,
    cell_size: int = CELL_SIZE,
) -> str:
    
    # Ridimensiona a 28x28 se necessario
    if cell_img.shape != (cell_size, cell_size):
        img = Image.fromarray(cell_img.astype(np.uint8)).resize(
            (cell_size, cell_size), Image.LANCZOS
        )
        cell_img = np.array(img)

    # Normalizza e appiattisce come durante il training
    feature_vector = (cell_img.astype(np.float32) / 255.0).flatten()
    return knn.predict([feature_vector])[0]


# -----------------------------------------------------------------------
# Entry point — eseguire questo file per generare il dataset e addestrare
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("Smart Drone Delivery — Componente ML")
    print("Classificatore KNN per riconoscimento simboli")
    print("=" * 50)

    # Step 1: Genera il dataset se non esiste
    if not DATASET_DIR.exists():
        generate_dataset()
    else:
        print(f"[ML] Dataset esistente trovato in '{DATASET_DIR}' — skip generazione")

    # Step 2: Addestra il KNN e valuta le prestazioni
    if not _SKLEARN_AVAILABLE:
        print("\n[ML] Errore: scikit-learn non è installato.")
        print("Installa con: pip install scikit-learn")
        sys.exit(1)

    knn = train_knn()

    # Step 3: Test rapido su un simbolo generato al volo
    print("\n[ML] Test rapido su simbolo generato:")
    for sym in SYMBOLS:
        vec  = _render_symbol(sym)
        pred = knn.predict([vec])[0]
        ok   = "OK" if pred == sym else f"ERRORE (predetto: {pred})"
        print(f"  Simbolo reale: {sym} -> Predetto: {pred} [{ok}]")
