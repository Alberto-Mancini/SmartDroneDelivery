
from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from environment import Grid
from sensors import DroneSensors


# ---------------------------------------------------------------------------
# Dizionario di abilitazione algoritmi
# ---------------------------------------------------------------------------

ENABLED_ALGORITHMS: dict[str, bool] = {
    # Imposta True per gli algoritmi da includere nel confronto (tasto C)
    "BFS":    False,
    "DFS":    False,
    "IDS":    False,
    "UCS":    False,
    "Greedy": False,
    "A*":     True,   # Algoritmo principale — ottimo e informato
    "IDA*":   False,
    "RBFS":   False,
    "SMA*":   False,
}


# ---------------------------------------------------------------------------
# Strutture dati per la ricerca
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchState:
    
    position:           tuple[int, int]
    pending_deliveries: frozenset        # Posizioni delle consegne non ancora effettuate
    steps:              int              # Numero di passi (per tie-breaking)

    def __lt__(self, other: "SearchState") -> bool:
        # Necessario per ordinamento in heapq quando f_cost e' uguale
        return self.steps < other.steps


@dataclass
class SearchNode:
   
    state:  SearchState
    parent: Optional["SearchNode"]
    action: Optional[tuple[int, int]]  # Posizione raggiunta con questa mossa
    g_cost: float                      # g(n): costo dal nodo iniziale
    h_cost: float = 0.0                # h(n): euristica (stima al goal)
    depth:  int   = 0                  # Profondita' nell'albero di ricerca

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def __lt__(self, other: "SearchNode") -> bool:
        # Necessario per heapq (priority queue)
        return self.f_cost < other.f_cost

    def path(self) -> list[tuple[int, int]]:
        """Ricostruisce il percorso risalendo la catena di nodi padre."""
        nodes, node = [], self
        while node is not None:
            if node.action is not None:
                nodes.append(node.action)
            node = node.parent
        nodes.reverse()
        return nodes

    def full_path(self, start: tuple[int, int]) -> list[tuple[int, int]]:
        """Restituisce il percorso completo incluso il nodo iniziale."""
        return [start] + self.path()


@dataclass
class SearchResult:
   
    algorithm:      str
    path:           list[tuple[int, int]]  # Sequenza di celle del percorso ottimo
    total_cost:     float                  # Costo totale del percorso
    nodes_expanded: int                    # Nodi espansi (misura del lavoro svolto)
    max_frontier:   int                    # Dimensione massima della frontiera (memoria)
    time_seconds:   float                  # Tempo di esecuzione
    success:        bool
    message:        str = ""

    def __str__(self) -> str:
        if not self.success:
            return f"[{self.algorithm}] FALLITO: {self.message}"
        return (f"[{self.algorithm}] "
                f"passi={len(self.path)-1} | costo={self.total_cost:.2f} | "
                f"nodi={self.nodes_expanded} | tempo={self.time_seconds*1000:.1f}ms")


# ---------------------------------------------------------------------------
# Controller principale — implementa tutti gli algoritmi
# ---------------------------------------------------------------------------

class SearchController:

    def __init__(self, grid: Grid, sensors: DroneSensors) -> None:
        self._grid    = grid
        self._sensors = sensors

    def _heuristic(self, pos: tuple[int, int], pending: frozenset, weight: float = 1.0) -> float:
        
        if not pending:
            # Tutte le consegne fatte: il drone deve solo tornare alla base
            return weight * self._grid.manhattan(pos, self._grid.start_pos)
        # Stima lower bound: raggiungi la piu' vicina, poi torna dalla piu' lontana
        nearest  = min(pending, key=lambda p: self._grid.manhattan(pos, p))
        farthest = max(pending, key=lambda p: self._grid.manhattan(pos, p))
        return weight * (self._grid.manhattan(pos, nearest)
                         + self._grid.manhattan(farthest, self._grid.start_pos))

    def _initial_state(self) -> SearchState:

        pending = frozenset(
            pos for pos in self._grid.delivery_positions
            if self._grid[pos].has_pending_delivery()
        )
        return SearchState(position=self._sensors.position, pending_deliveries=pending, steps=0)

    def _is_goal(self, state: SearchState) -> bool:
    
        return not state.pending_deliveries and state.position == self._grid.start_pos

    def _successors(self, node: SearchNode) -> list[tuple[tuple[int, int], float, SearchState]]:
        state, pos = node.state, node.state.position
        result = []
        # Mosse cardinali: Nord, Sud, Est, Ovest
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            new_pos = (pos[0] + dr, pos[1] + dc)
            r, c = new_pos
            # Controlla bounds e passabilita'
            if not self._grid.in_bounds(r, c) or not self._grid.is_passable(r, c):
                continue
            step_cost   = self._grid.step_cost(r, c)
            # Aggiorna il set delle consegne rimuovendo la posizione appena raggiunta
            new_pending = set(state.pending_deliveries)
            new_pending.discard(new_pos)  # Se e' una DELIVERY, la rimuove
            new_state = SearchState(
                position=new_pos,
                pending_deliveries=frozenset(new_pending),
                steps=state.steps + 1,
            )
            result.append((new_pos, step_cost, new_state))
        return result

    @staticmethod
    def _timer() -> float:
        return time.perf_counter()

    def _make_result(self, algo, node, start_pos, expanded, max_f, t0):
        """Costruisce un SearchResult dal nodo goal (o None se fallito)."""
        elapsed = self._timer() - t0
        if node is None:
            return SearchResult(algo, [], 0.0, expanded, max_f, elapsed, False, "nessun percorso trovato")
        return SearchResult(algo, node.full_path(start_pos), node.g_cost, expanded, max_f, elapsed, True)

    # -----------------------------------------------------------------------
    # BFS — Breadth-First Search (lec02)
    # -----------------------------------------------------------------------

    def bfs(self) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        root = SearchNode(start, None, None, 0.0)
        # Frontiera FIFO — nodi da espandere in ordine di inserimento
        frontier, visited = deque([root]), {start}
        expanded, max_f = 0, 1
        while frontier:
            max_f = max(max_f, len(frontier))
            node = frontier.popleft()  # FIFO: prende dal fronte della coda
            expanded += 1
            if self._is_goal(node.state):
                return self._make_result("BFS", node, start.position, expanded, max_f, t0)
            for new_pos, cost, new_state in self._successors(node):
                if new_state not in visited:
                    visited.add(new_state)
                    frontier.append(SearchNode(new_state, node, new_pos, node.g_cost + cost, depth=node.depth + 1))
        return self._make_result("BFS", None, start.position, expanded, max_f, t0)

    # -----------------------------------------------------------------------
    # DFS — Depth-First Search (lec02)
    # -----------------------------------------------------------------------

    def dfs(self, max_depth: int = 200) -> SearchResult:
        
        t0, start = self._timer(), self._initial_state()
        # Stack LIFO — nodi piu' recenti esplorati per primi
        frontier, visited = [SearchNode(start, None, None, 0.0)], set()
        expanded, max_f = 0, 1
        while frontier:
            max_f = max(max_f, len(frontier))
            node = frontier.pop()  # LIFO: prende dall'ultimo inserito
            if node.state in visited:
                continue
            visited.add(node.state)
            expanded += 1
            if self._is_goal(node.state):
                return self._make_result("DFS", node, start.position, expanded, max_f, t0)
            # Espande solo se non ha raggiunto il limite di profondita'
            if node.depth < max_depth:
                for new_pos, cost, new_state in self._successors(node):
                    if new_state not in visited:
                        frontier.append(SearchNode(new_state, node, new_pos, node.g_cost + cost, depth=node.depth + 1))
        return self._make_result("DFS", None, start.position, expanded, max_f, t0)

    # -----------------------------------------------------------------------
    # IDS — Iterative Deepening Search (lec02)
    # -----------------------------------------------------------------------

    def ids(self, max_depth: int = 200) -> SearchResult:
        t0, start, total_exp = self._timer(), self._initial_state(), 0
        # Aumenta progressivamente il limite di profondita'
        for limit in range(max_depth + 1):
            result, exp, _ = self._dls(start, limit)
            total_exp += exp
            if result is not None:
                return self._make_result("IDS", result, start.position, total_exp, 0, t0)
        return self._make_result("IDS", None, start.position, total_exp, 0, t0)

    def _dls(self, start, limit):
        root = SearchNode(start, None, None, 0.0)
        stack, expanded, max_f = [(root, set())], 0, 1
        while stack:
            max_f = max(max_f, len(stack))
            node, path_states = stack.pop()
            expanded += 1
            if self._is_goal(node.state):
                return node, expanded, max_f
            if node.depth < limit:
                for new_pos, cost, new_state in self._successors(node):
                    if new_state not in path_states:
                        # path_states evita cicli lungo il percorso corrente
                        stack.append((SearchNode(new_state, node, new_pos, node.g_cost + cost, depth=node.depth + 1),
                                      path_states | {new_state}))
        return None, expanded, max_f

    # -----------------------------------------------------------------------
    # UCS — Uniform Cost Search (lec02)
    # -----------------------------------------------------------------------

    def ucs(self) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        # Priority queue ordinata per costo cumulativo g(n)
        frontier, visited = [(0.0, SearchNode(start, None, None, 0.0))], {}
        expanded, max_f = 0, 1
        while frontier:
            max_f = max(max_f, len(frontier))
            g, node = heapq.heappop(frontier)  # Estrae il nodo a costo minore
            # Salta se abbiamo gia' trovato un percorso migliore per questo stato
            if node.state in visited and visited[node.state] <= g:
                continue
            visited[node.state] = g
            expanded += 1
            if self._is_goal(node.state):
                return self._make_result("UCS", node, start.position, expanded, max_f, t0)
            for new_pos, cost, new_state in self._successors(node):
                new_g = node.g_cost + cost
                if new_state not in visited or visited[new_state] > new_g:
                    heapq.heappush(frontier, (new_g, SearchNode(new_state, node, new_pos, new_g, depth=node.depth + 1)))
        return self._make_result("UCS", None, start.position, expanded, max_f, t0)

    # -----------------------------------------------------------------------
    # Greedy Best-First Search (lec04)
    # -----------------------------------------------------------------------

    def greedy(self) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        h0 = self._heuristic(start.position, start.pending_deliveries)
        # Priority queue ordinata solo per h(n) — nessun costo backward
        frontier, visited = [(h0, SearchNode(start, None, None, 0.0, h0))], set()
        expanded, max_f = 0, 1
        while frontier:
            max_f = max(max_f, len(frontier))
            _, node = heapq.heappop(frontier)
            if node.state in visited:
                continue
            visited.add(node.state)
            expanded += 1
            if self._is_goal(node.state):
                return self._make_result("Greedy", node, start.position, expanded, max_f, t0)
            for new_pos, cost, new_state in self._successors(node):
                if new_state not in visited:
                    h = self._heuristic(new_state.position, new_state.pending_deliveries)
                    heapq.heappush(frontier, (h, SearchNode(new_state, node, new_pos, node.g_cost + cost, h, node.depth + 1)))
        return self._make_result("Greedy", None, start.position, expanded, max_f, t0)

    # -----------------------------------------------------------------------
    # A* Search (lec04) — algoritmo principale
    # -----------------------------------------------------------------------

    def astar(self, weight: float = 1.0) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        h0   = self._heuristic(start.position, start.pending_deliveries, weight)
        root = SearchNode(start, None, None, 0.0, h0)
        # Priority queue ordinata per f(n) = g(n) + h(n)
        frontier, best_g = [(root.f_cost, root)], {}
        expanded, max_f = 0, 1
        while frontier:
            max_f = max(max_f, len(frontier))
            _, node = heapq.heappop(frontier)
            # Graph search: salta se abbiamo gia' espanso con costo minore
            if node.state in best_g and best_g[node.state] < node.g_cost:
                continue
            best_g[node.state] = node.g_cost
            expanded += 1
            if self._is_goal(node.state):
                return self._make_result("A*", node, start.position, expanded, max_f, t0)
            for new_pos, cost, new_state in self._successors(node):
                new_g = node.g_cost + cost
                if new_state not in best_g or best_g[new_state] > new_g:
                    h = self._heuristic(new_state.position, new_state.pending_deliveries, weight)
                    heapq.heappush(frontier, (new_g + h, SearchNode(new_state, node, new_pos, new_g, h, node.depth + 1)))
        return self._make_result("A*", None, start.position, expanded, max_f, t0)

    # -----------------------------------------------------------------------
    # IDA* — Iterative Deepening A* (lec06)
    # -----------------------------------------------------------------------

    def idastar(self) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        h0   = self._heuristic(start.position, start.pending_deliveries)
        root = SearchNode(start, None, None, 0.0, h0)
        # Soglia iniziale = euristica dello stato iniziale
        threshold, total_exp = h0, 0
        while True:
            result, exp, next_t = self._ida_search(root, threshold, set())
            total_exp += exp
            if result is not None:
                return self._make_result("IDA*", result, start.position, total_exp, 0, t0)
            if next_t == float("inf"):
                return self._make_result("IDA*", None, start.position, total_exp, 0, t0)
            # Nuova soglia = minimo f(n) che ha superato la soglia precedente
            threshold = next_t

    def _ida_search(self, node, threshold, path_states):
        if node.f_cost > threshold:
            # Nodo oltre soglia: restituisce il suo f come candidato per la prossima iterazione
            return None, 0, node.f_cost
        if self._is_goal(node.state):
            return node, 1, threshold
        min_exceeded, expanded = float("inf"), 1
        path_states = path_states | {node.state}  # Evita cicli nel percorso corrente
        for new_pos, cost, new_state in self._successors(node):
            if new_state in path_states:
                continue
            h = self._heuristic(new_state.position, new_state.pending_deliveries)
            child = SearchNode(new_state, node, new_pos, node.g_cost + cost, h, node.depth + 1)
            result, sub_exp, sub_t = self._ida_search(child, threshold, path_states)
            expanded += sub_exp
            if result is not None:
                return result, expanded, threshold
            min_exceeded = min(min_exceeded, sub_t)
        return None, expanded, min_exceeded

    # -----------------------------------------------------------------------
    # RBFS — Recursive Best-First Search (lec06)
    # -----------------------------------------------------------------------

    def rbfs(self) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        h0   = self._heuristic(start.position, start.pending_deliveries)
        root = SearchNode(start, None, None, 0.0, h0)
        counter = [0]  # Lista mutabile per aggiornamento in closure ricorsiva
        result, _ = self._rbfs_search(root, float("inf"), set(), counter)
        return self._make_result("RBFS", result, start.position, counter[0], 0, t0)

    def _rbfs_search(self, node, f_limit, path_states, counter):
        if self._is_goal(node.state):
            return node, node.f_cost
        successors = []
        path_states = path_states | {node.state}
        for new_pos, cost, new_state in self._successors(node):
            if new_state in path_states:
                continue
            h = self._heuristic(new_state.position, new_state.pending_deliveries)
            g = node.g_cost + cost
            # max(g+h, node.f_cost): propaga il costo del padre se maggiore
            successors.append((max(g + h, node.f_cost), SearchNode(new_state, node, new_pos, g, h, node.depth + 1)))
        if not successors:
            return None, float("inf")
        while True:
            successors.sort(key=lambda x: x[0])
            best_f, best = successors[0]
            # Se il miglior nodo supera il limite, torna al livello superiore
            if best_f > f_limit:
                return None, best_f
            # Alternativa: secondo nodo migliore (usato come nuovo f_limit)
            alt_f = successors[1][0] if len(successors) > 1 else float("inf")
            counter[0] += 1
            result, best_f_new = self._rbfs_search(best, min(f_limit, alt_f), path_states, counter)
            successors[0] = (best_f_new, best)  # Aggiorna il costo del ramo esplorato
            if result is not None:
                return result, best_f_new

    # -----------------------------------------------------------------------
    # SMA* — Simplified Memory-Bounded A* (lec06)
    # -----------------------------------------------------------------------

    def smastar(self, memory_limit: int = 500) -> SearchResult:
        t0, start = self._timer(), self._initial_state()
        h0      = self._heuristic(start.position, start.pending_deliveries)
        root    = SearchNode(start, None, None, 0.0, h0)
        counter = 0  # Tie-breaker per ordinamento stabile nella priority queue
        frontier, in_frontier = [(root.f_cost, counter, root)], {start: root.f_cost}
        expanded, max_f = 0, 1
        while frontier:
            max_f = max(max_f, len(frontier))
            _, _, node = heapq.heappop(frontier)
            in_frontier.pop(node.state, None)
            expanded += 1
            if self._is_goal(node.state):
                return self._make_result("SMA*", node, start.position, expanded, max_f, t0)
            for new_pos, cost, new_state in self._successors(node):
                new_g = node.g_cost + cost
                h     = self._heuristic(new_state.position, new_state.pending_deliveries)
                f     = new_g + h
                # Salta se esiste gia' un percorso migliore per questo stato
                if new_state in in_frontier and in_frontier[new_state] <= f:
                    continue
                counter += 1
                child = SearchNode(new_state, node, new_pos, new_g, h, node.depth + 1)
                heapq.heappush(frontier, (f, counter, child))
                in_frontier[new_state] = f
                # Se supera il limite di memoria: rimuovi il nodo peggiore
                if len(frontier) > memory_limit:
                    worst_idx = max(range(len(frontier)), key=lambda i: frontier[i][0])
                    worst_state = frontier[worst_idx][2].state
                    frontier[worst_idx] = frontier[-1]
                    frontier.pop()
                    heapq.heapify(frontier)
                    in_frontier.pop(worst_state, None)
        return self._make_result("SMA*", None, start.position, expanded, max_f, t0)

    # -----------------------------------------------------------------------
    # Confronto di tutti gli algoritmi abilitati
    # -----------------------------------------------------------------------

    def compare_all(self, verbose: bool = True) -> dict[str, SearchResult]:
        algo_map = {
            "BFS": self.bfs, "DFS": self.dfs, "IDS": self.ids,
            "UCS": self.ucs, "Greedy": self.greedy, "A*": self.astar,
            "IDA*": self.idastar, "RBFS": self.rbfs, "SMA*": self.smastar,
        }
        results = {}
        for name, fn in algo_map.items():
            if not ENABLED_ALGORITHMS.get(name, False):
                if verbose:
                    print(f"[{name}] disabilitato")
                continue
            # Reset necessario per garantire lo stesso stato iniziale per ogni algoritmo
            self._grid.reset_deliveries()
            try:
                res = fn()
            except RecursionError:
                res = SearchResult(name, [], 0.0, 0, 0, 0.0, False, "RecursionError")
            results[name] = res
            if verbose:
                print(res)
        return results
