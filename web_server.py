"""
Web Server Flask per visualizzazione Smart Drone Delivery su WSL
- Griglia interattiva della mappa
- Dashboard ML con metriche
- Simulazione pathfinding animato
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
from pathlib import Path
import threading
import time

from environment import Grid, CellType
from image_parser import parse_image, generate_test_image
from sensors import DroneSensors
from movement import MovementEngine
from controller import SearchController, SearchResult, ENABLED_ALGORITHMS
from ml_classifier import train_knn, load_model

import traceback

app = Flask(__name__)
CORS(app)

# Stato globale della simulazione
class SimulationState:
    def __init__(self):
        self.grid = None
        self.sim_path = []
        self.step_index = 0
        self.paused = True
        self.finished = False
        self.algo_name = "A*"
        self.speed = 2.0
        self.result = None
        self.log = []
        
    def reset(self):
        self.step_index = 0
        self.paused = True
        self.finished = False
        self.log = []

sim_state = SimulationState()

# ========== ENDPOINTS ==========

@app.route('/')
def index():
    """Pagina principale"""
    return render_template('index.html')

@app.route('/api/grid')
def get_grid():
    """Carica e restituisce griglia come JSON"""
    if sim_state.grid is None:
        try:
            # Genera griglia di test
            test_img = generate_test_image("test_map.png")
            print(f"[DEBUG] Immagine generata: {test_img}")
            sim_state.grid = parse_image(test_img)
            print(f"[DEBUG] Griglia caricata: {sim_state.grid.rows}x{sim_state.grid.cols}")
            print(f"[DEBUG] START pos: {sim_state.grid.start_pos}")
        except Exception as e:
            print(f"[ERROR] get_grid: {e}")
            traceback.print_exc()
            raise
    
    grid_data = {
        'rows': sim_state.grid.rows,
        'cols': sim_state.grid.cols,
        'start_pos': sim_state.grid.start_pos,
        'delivery_positions': sim_state.grid.delivery_positions,
        'cells': []
    }
    
    # Serializza le celle
    for r in range(sim_state.grid.rows):
        row = []
        for c in range(sim_state.grid.cols):
            cell = sim_state.grid[r, c]
            row.append({
                'type': cell.cell_type.value,
                'type_name': cell.cell_type.name,
                'delivered': cell.delivered
            })
        grid_data['cells'].append(row)
    
    return jsonify(grid_data)

@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    """Esegui simulazione con algoritmo selezionato"""
    try:
        data = request.json
        algo = data.get('algorithm', 'A*')
        
        sim_state.algo_name = algo
        sim_state.reset()
        
        # Assicurati che la griglia sia caricata
        if sim_state.grid is None:
            test_img = generate_test_image("test_map.png")
            sim_state.grid = parse_image(test_img)
        
        # Esegui ricerca
        controller = SearchController(sim_state.grid, DroneSensors(sim_state.grid, sim_state.grid.start_pos))
        algos = {
            'BFS': controller.bfs, 'DFS': controller.dfs, 'IDS': controller.ids,
            'UCS': controller.ucs, 'Greedy': controller.greedy, 'A*': controller.astar,
            'IDA*': controller.idastar, 'RBFS': controller.rbfs, 'SMA*': controller.smastar
        }
        
        result = algos[algo]()
        
        sim_state.result = result
        sim_state.sim_path = result.path if result.success else [sim_state.grid.start_pos]
        sim_state.step_index = 0
        sim_state.log.append(f"Algoritmo: {algo}")
        
        return jsonify({
            'success': result.success,
            'path': result.path,
            'nodes_expanded': result.nodes_expanded,
            'time_ms': result.time_seconds * 1000,
            'total_cost': result.total_cost,
            'message': result.message
        })
    except Exception as e:
        print(f"[ERROR] simulate: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'path': [],
            'nodes_expanded': 0,
            'time_ms': 0,
            'total_cost': 0,
            'message': str(e)
        }), 500

@app.route('/api/step', methods=['POST'])
def advance_step():
    """Avanza di uno step nella simulazione"""
    if not sim_state.paused and sim_state.step_index + 1 < len(sim_state.sim_path):
        sim_state.step_index += 1
        pos = sim_state.sim_path[sim_state.step_index]
        r, c = pos
        
        # Segna consegna se presente
        if sim_state.grid[r, c].has_pending_delivery():
            sim_state.grid.mark_delivered(r, c)
            sim_state.log.append(f"Consegna {pos} OK")
    elif sim_state.step_index + 1 >= len(sim_state.sim_path):
        sim_state.finished = True
    
    return jsonify({
        'step_index': sim_state.step_index,
        'current_pos': sim_state.sim_path[sim_state.step_index] if sim_state.step_index < len(sim_state.sim_path) else None,
        'finished': sim_state.finished,
        'log': sim_state.log[-5:]
    })

@app.route('/api/state')
def get_state():
    """Ottieni stato attuale simulazione"""
    return jsonify({
        'step_index': sim_state.step_index,
        'total_steps': len(sim_state.sim_path) - 1 if sim_state.sim_path else 0,
        'current_pos': sim_state.sim_path[sim_state.step_index] if sim_state.sim_path and sim_state.step_index < len(sim_state.sim_path) else None,
        'paused': sim_state.paused,
        'finished': sim_state.finished,
        'algorithm': sim_state.algo_name,
        'speed': sim_state.speed
    })

@app.route('/api/pause', methods=['POST'])
def toggle_pause():
    """Pausa/Riprendi simulazione"""
    sim_state.paused = not sim_state.paused
    return jsonify({'paused': sim_state.paused})

@app.route('/api/speed', methods=['POST'])
def set_speed():
    """Cambia velocità simulazione"""
    data = request.json
    sim_state.speed = max(0.5, min(20.0, data.get('speed', 2.0)))
    return jsonify({'speed': sim_state.speed})

@app.route('/api/ml-metrics')
def get_ml_metrics():
    """Restituisce metriche ML dal classificatore"""
    try:
        model = load_model()
        # TODO: calcolare metriche se necessario
        return jsonify({
            'status': 'Model loaded',
            'accuracy': 1.0  # Placeholder
        })
    except:
        return jsonify({'error': 'Model not found'}), 404

if __name__ == '__main__':
    print("🚀 Starting Smart Drone Delivery Web Server...")
    print("📍 Open http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')