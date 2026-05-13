# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment import Grid


@dataclass
class SensorState:

    position:    tuple[int, int]  # Posizione corrente (riga, colonna)
    steps_taken: int = 0          # Numero di passi effettuati dall'inizio

    def clone(self) -> "SensorState":
        """Crea una copia indipendente dello stato."""
        return SensorState(position=self.position, steps_taken=self.steps_taken)


class DroneSensors:
    """
    Gestisce i sensori del drone: posizione e rilevamento vento.
    """

    def __init__(self, grid: "Grid", start_pos: tuple[int, int]) -> None:
        self._grid       = grid
        self.position    = start_pos   # Posizione iniziale del drone
        self.steps_taken = 0           # Contatore passi

    def wind_at(self, row: int, col: int) -> bool:
        """
        Rileva se la cella (row, col) e' una zona di vento.
        Il vento raddoppia il costo di attraversamento (WIND_MULTIPLIER = 2.0).
        """
        from environment import CellType
        return self._grid[row, col].cell_type == CellType.WIND

    def wind_multiplier_at(self, row: int, col: int) -> float:
        """Restituisce il moltiplicatore di costo per la cella (1.0 o 2.0 se WIND)."""
        from environment import WIND_MULTIPLIER
        return WIND_MULTIPLIER if self.wind_at(row, col) else 1.0

    def update(self, new_pos: tuple[int, int]) -> dict:
        """
        Aggiorna i sensori dopo uno spostamento.
        Incrementa il contatore passi e aggiorna la posizione.
        Restituisce un dizionario con i dati sensoriali correnti.
        """
        self.position = new_pos
        self.steps_taken += 1
        r, c = new_pos
        return {
            "position":    new_pos,
            "steps_taken": self.steps_taken,
            "wind":        self.wind_at(r, c),  # True se il drone e' in una cella WIND
        }

    def get_state(self) -> SensorState:
        """Salva lo stato corrente dei sensori (usato per confronti tra algoritmi)."""
        return SensorState(position=self.position, steps_taken=self.steps_taken)

    def set_state(self, state: SensorState) -> None:
        """Ripristina uno stato precedentemente salvato."""
        self.position    = state.position
        self.steps_taken = state.steps_taken

    def status(self) -> str:
        return f"[Sensors] pos={self.position} | steps={self.steps_taken}"

    def __repr__(self) -> str:
        return self.status()
