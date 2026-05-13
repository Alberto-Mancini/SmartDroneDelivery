
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from environment import Grid
    from sensors import DroneSensors


class Direction(Enum):
    NORTH = (-1,  0)   # Riga diminuisce
    SOUTH = ( 1,  0)   # Riga aumenta
    EAST  = ( 0,  1)   # Colonna aumenta
    WEST  = ( 0, -1)   # Colonna diminuisce

    def apply(self, pos: tuple[int, int]) -> tuple[int, int]:
        """Applica la direzione a una posizione e restituisce la nuova posizione."""
        return (pos[0] + self.value[0], pos[1] + self.value[1])

    def __str__(self) -> str:
        return {Direction.NORTH: "^", Direction.SOUTH: "v",
                Direction.EAST:  ">", Direction.WEST:  "<"}[self]

ALL_DIRECTIONS = list(Direction)  # Lista di tutte le direzioni possibili


@dataclass
class MoveResult:
    success:       bool
    old_pos:       tuple[int, int]   # Posizione prima della mossa
    new_pos:       tuple[int, int]   # Posizione dopo la mossa
    direction:     Direction
    steps_taken:   int  = 0          # Totale passi effettuati fino ad ora
    delivered:     bool = False      # True se e' avvenuta una consegna
    delivery_type: Optional[str] = None
    is_wind:       bool = False      # True se la nuova cella e' WIND
    is_start:      bool = False      # True se il drone e' tornato alla base
    fail_reason:   str  = ""         # Motivo del fallimento (se success=False)

    def __str__(self) -> str:
        if not self.success:
            return f"Mossa FALLITA ({self.fail_reason})"
        tag = ""
        if self.delivered: tag += f" [CONSEGNA {self.delivery_type}]"
        if self.is_wind:   tag += " [VENTO]"
        return f"{self.direction} {self.old_pos}->{self.new_pos} | step={self.steps_taken}{tag}"


class MovementEngine:
    
    def __init__(self, grid: "Grid", sensors: "DroneSensors") -> None:
        self._grid    = grid
        self._sensors = sensors

    def valid_moves(
        self, pos: Optional[tuple[int, int]] = None
    ) -> list[tuple[Direction, tuple[int, int]]]:

        if pos is None:
            pos = self._sensors.position
        result = []
        for direction in ALL_DIRECTIONS:
            new_pos = direction.apply(pos)
            r, c = new_pos
            if self._grid.in_bounds(r, c) and self._grid.is_passable(r, c):
                result.append((direction, new_pos))
        return result

    def valid_positions(self, pos: Optional[tuple[int, int]] = None) -> list[tuple[int, int]]:
        """Restituisce solo le posizioni (senza direzione) delle mosse valide."""
        return [p for _, p in self.valid_moves(pos)]

    def execute_move(
        self,
        direction: Direction,
        pos: Optional[tuple[int, int]] = None,
    ) -> MoveResult:
        
        from environment import CellType
        if pos is None:
            pos = self._sensors.position
        old_pos = pos
        new_pos = direction.apply(pos)
        r, c = new_pos

        # Verifica bounds e passabilita'
        if not self._grid.in_bounds(r, c):
            return MoveResult(False, old_pos, new_pos, direction, fail_reason="fuori griglia")
        if not self._grid.is_passable(r, c):
            return MoveResult(False, old_pos, new_pos, direction, fail_reason="cella non attraversabile (X)")

        # Aggiorna i sensori con la nuova posizione
        sensor_data   = self._sensors.update(new_pos)
        cell          = self._grid[r, c]
        delivered     = False
        delivery_type = None

        # Effettua la consegna se la cella e' un punto di consegna pendente
        if cell.has_pending_delivery():
            delivered = self._grid.mark_delivered(r, c)
            if delivered:
                delivery_type = cell.cell_type.name

        return MoveResult(
            success       = True,
            old_pos       = old_pos,
            new_pos       = new_pos,
            direction     = direction,
            steps_taken   = sensor_data["steps_taken"],
            delivered     = delivered,
            delivery_type = delivery_type,
            is_wind       = sensor_data["wind"],
            is_start      = cell.cell_type == CellType.START,
        )

    def execute_path(self, path: list[tuple[int, int]], verbose: bool = False) -> list[MoveResult]:
        """
        Esegue una sequenza di posizioni come percorso.
        Determina automaticamente la direzione tra posizioni consecutive.
        """
        results = []
        for i in range(1, len(path)):
            prev, curr = path[i - 1], path[i]
            dr, dc = curr[0] - prev[0], curr[1] - prev[1]
            direction = next((d for d in ALL_DIRECTIONS if d.value == (dr, dc)), None)
            if direction is None:
                raise ValueError(f"Passo non valido: {prev} -> {curr}")
            result = self.execute_move(direction, pos=prev)
            results.append(result)
            if verbose:
                print(result)
            if not result.success:
                break
        return results

    def path_cost(self, path: list[tuple[int, int]]) -> float:
        """Calcola il costo totale di un percorso sommando i costi di ogni passo."""
        return sum(self._grid.step_cost(r, c) for r, c in path[1:])

    def status(self) -> str:
        moves = self.valid_moves()
        return (f"[Movement] pos={self._sensors.position} | "
                f"mosse valide: {len(moves)} "
                f"({', '.join(str(d) for d, _ in moves)})")
