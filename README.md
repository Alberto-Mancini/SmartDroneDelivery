# Smart Drone Delivery 🚁

Sistema di **simulazione e ottimizzazione** per consegne tramite droni in griglia, con confronto tra 9 algoritmi di ricerca intelligenti.

## ✨ Caratteristiche

- 🗺️ **Griglia dinamica** con ostacoli, zone di vento (costo 2x), punti di consegna multi-destinazione
- 🤖 **9 Algoritmi di ricerca**: BFS, DFS, IDS, UCS, Greedy, A*, IDA*, RBFS, SMA*
- 🧠 **Classificatore ML** (KNN) per riconoscimento celle da immagini con dataset 1000 campioni
- 🎮 **Interfaccia pygame** interattiva con visualizzazione real-time del drone
- 📊 **Benchmark automatico** per confrontare performance e costi tra algoritmi
- 💨 **Sensori drone** per rilevamento vento e tracciamento consegne completate
- ⚡ **Multithreading** con timeout per algoritmi slow (IDA*, SMA*)

## 📋 Requisiti

- Python 3.11+
- pygame
- numpy
- scikit-learn
- pillow

## 🚀 Installazione & Setup

### 1. Setup ambiente virtuale
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Training classificatore (primo avvio)
```bash
python ml_classifier.py
```
Output atteso:
- Genera dataset 1000 immagini (5 classi, dual-mode: 50% puliti + 50% perturbati)
- Allena KNN con train/test split 80/20
- Salva modello in `knn_model.pkl`
- Accuracy: ~99%

### 3. Lanciare simulazione
```bash
python main.py
```

Opzionale: passare immagine custom
```bash
python main.py path/to/custom_map.png
```

## 🎮 Controlli in-game

| Tasto | Azione |
|-------|--------|
| **SPACE** | Pausa/Avvia simulazione |
| **R** | Ripristina (stesso algoritmo) |
| **1-9** | Seleziona algoritmo (A* default) |
| **+/-** | Aumenta/Diminuisci velocità |
| **C** | Benchmark: confronta algoritmi abilitati |
| **ESC/Q** | Esci |

## 📁 Struttura Progetto

```
SmartDroneDelivery/
├── main.py                    # Loop principale pygame, renderer
├── environment.py             # Griglia, celle, logica costi e vento
├── controller.py              # Implementazione 9 algoritmi di ricerca
├── sensors.py                 # Sensori drone (posizione, vento, consegne)
├── movement.py                # Motore movimento e validazione spostamenti
├── image_parser.py            # Parser immagini → griglia con ML
├── ml_classifier.py           # Training/predizione KNN
├── dataset/                   # Training data (1000 campioni, 5 classi: S,D,W,X,E)
│   ├── S/  (START)
│   ├── D/  (DELIVERY)
│   ├── W/  (WIND)
│   ├── X/  (BLOCKED)
│   └── E/  (EMPTY)
├── requirements.txt           # Dipendenze Python (numpy, pillow, scikit-learn, pygame)
├── .gitignore                 # Esclude venv, cache, modelli generati
├── knn_model.pkl              # Modello addestrato (generato)
├── test_map.png               # Mappa di test (generata)
├── LICENSE                    # MIT License
├── README.md                  # Questo file
├── DEBUGGING_LOG.md           # Log debugging e risoluzione problemi
└── ML_MODEL_FIX.md            # Analisi e fix del classificatore ML
```

## 🧠 Algoritmi Implementati

### Ricerca Non-Informata
- **BFS** (Breadth-First Search) → Espansione per livelli, garantisce soluzione minima
- **DFS** (Depth-First Search) → Espansione in profondità, memoria minima
- **IDS** (Iterative Deepening Search) → Combina vantaggi BFS+DFS

### Ricerca Ottimale per Costo
- **UCS** (Uniform Cost Search) → Percorso a costo minimo, rispetta moltiplicatore vento
- **A*** → Informata con euristica Manhattan, **più veloce su multi-consegna**

### Ricerca Greedy/Approssimata
- **Greedy** → Solo euristica, veloce ma non ottimale

### Ricerca con Vincoli di Memoria
- **IDA*** → A* iterativo, memoria lineare ma ricalcola nodi
- **RBFS** (Recursive Best-First) → Ricorsivo, memoria lineare
- **SMA*** (Simplified Memory-Bounded A*) → Memoria limitata

## 📊 Benchmark

Premi **C** in-game per confrontare tutti gli algoritmi abilitati:

```
[ML] Dataset: 1000 campioni, 5 classi (dual-mode: 50% puliti + 50% perturbati)
[ML] Accuracy: 0.9950

CONFRONTO ALGORITMI ABILITATI
================================================
A*: 12 nodi | 45.2ms | Costo: 24.5 | Passi: 18
BFS: 34 nodi | 102.1ms | Costo: 24.5 | Passi: 18
...
```

## 🤖 Classificatore ML

- **Algoritmo**: K-Nearest Neighbors (k=3, distanza euclidea)
- **Dataset**: 1000 immagini 28x28 (200 per classe, dual-mode: 50% puliti + 50% con perturbazioni)
- **Preprocessing**: Normalizzazione [0,1], appiattimento a 784 feature
- **Training**: 800 campioni (80%)
- **Test**: 200 campioni (20%)
- **Accuracy**: 99.5% (alta robustezza grazie al dual-mode dataset)

## 🎯 Strategie di Testing

### Test rapido (default)
```bash
python main.py
```
Genera mappa 8x8 con simboli random

### Test con algoritmo specifico
1. Lanciare
2. Premere numero 1-9 per cambiar algoritmo
3. Premere SPACE per avviare

### Benchmark completo
1. Abilitare algoritmi in `controller.py` (ENABLED_ALGORITHMS)
2. Premere **C** in-game
3. Vedi confronto tempo/nodi/costo

## 🔧 Configurazione

### File `controller.py`
```python
ENABLED_ALGORITHMS: dict[str, bool] = {
    "BFS": False,
    "A*": True,  # ← default per confronti
    ...
}
```

### File `ml_classifier.py`
```python
CELL_SIZE = 28          # Pixel per cella (compatibile dataset)
SAMPLES_CLASS = 200     # Campioni per classe
K_NEIGHBORS = 3         # Vicinanza KNN
```

## 📈 Performance Attese

- **Griglia 8x8** con 2-3 consegne → A* ~10-20ms
- **BFS/DFS** → più lenti ma garantiti
- **Timeout 5s** per algoritmi slow (IDA*, SMA*, RBFS)
- **FPS**: 60 (pygame main loop)

## 🐛 Troubleshooting

### "Nessuna cella START trovata"
→ Classificatore ML non allena correttamente. Esegui:
```bash
rm -rf dataset/ knn_model.pkl
python ml_classifier.py
```

### pygame non trovato
```bash
pip install pygame
```

### RecursionError su IDA*/RBFS
→ Aumento limite ricorsione in `main.py`:
```python
sys.setrecursionlimit(50000)
```

## 📝 Commit Message Format (convenzione)

```
feat:    Nuova funzionalità
fix:     Bug fix
docs:    Documentazione
refactor: Refactoring codice
test:    Test
perf:    Performance improvement
```

Esempio:
```bash
git commit -m "feat: Add SMA* algorithm with memory limit"
```

## 🌳 Branch Strategy

```
main/              # Stabile, sempre deployable
├── feature/*      # Nuove feature (es. feature/new-algorithms)
├── bugfix/*       # Bug fix (es. bugfix/classifier-accuracy)
└── experiment/*   # Test sperimentali (es. experiment/gpu-support)
```

Workflow:
```bash
git checkout -b feature/my-feature
# ... lavora ...
git push origin feature/my-feature
# → Apri PR su GitHub
```

## 📄 Licenza

MIT License - © 2026 Alberto Mancini

Questo progetto è distribuito sotto licenza MIT. Vedi il file [LICENSE](LICENSE) per i dettagli.
