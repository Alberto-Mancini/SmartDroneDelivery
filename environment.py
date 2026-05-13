
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import numpy as np


class CellType(IntEnum):
    EMPTY    = 0   # Cella libera — attraversabile a costo base 1.0
    START    = 1   # Posizione di partenza e ritorno del drone
    BLOCKED  = 3   # Ostacolo invalicabile (costo infinito)
    DELIVERY = 4   # Punto di consegna — obiettivo parziale da raggiungere
    WIND     = 5   # Zona di vento — costo raddoppiato (x2)


BASE_COST: dict[CellType, float] = {
    CellType.EMPTY:    1.0,
    CellType.START:    1.0,    # Tornare alla base ha costo normale
    CellType.BLOCKED:  float("inf"),  # Invalicabile: costo infinito
    CellType.DELIVERY: 1.0,
    CellType.WIND:     1.0,    # Il moltiplicatore viene applicato in step_cost
}

# Moltiplicatore applicato al costo nelle celle WIND (vincolo del progetto)
WIND_MULTIPLIER = 2.0


@dataclass
class GridCell:
    cell_type: CellType
    delivered: bool = False  # True se la consegna in questa cella e' gia' avvenuta

    def is_passable(self) -> bool:
        """Restituisce True se il drone puo' entrare nella cella (non e' BLOCKED)."""
        return self.cell_type != CellType.BLOCKED

    def has_pending_delivery(self) -> bool:
        """Restituisce True se la cella e' un punto di consegna non ancora servito."""
        return self.cell_type == CellType.DELIVERY and not self.delivered

    def __repr__(self) -> str:
        return f"GridCell({self.cell_type.name})"


class Grid:
    def __init__(self, cells: np.ndarray) -> None:
        if cells.ndim != 2:
            raise ValueError("L'array celle deve essere 2D.")
        self.rows, self.cols = cells.shape
        self.cells: np.ndarray = cells

        # Individua le posizioni speciali nella griglia al momento della creazione
        self.start_pos          = self._find_unique(CellType.START, required=True)
        self.delivery_positions = self._find_all(CellType.DELIVERY)

    def __getitem__(self, pos: tuple[int, int]) -> GridCell:
        """Accesso diretto a una cella tramite (riga, colonna)."""
        return self.cells[pos[0], pos[1]]

    def in_bounds(self, row: int, col: int) -> bool:
        """Controlla se le coordinate sono dentro i limiti della griglia."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_passable(self, row: int, col: int) -> bool:
        """Controlla se la cella e' dentro la griglia e non e' BLOCKED."""
        return self.in_bounds(row, col) and self.cells[row, col].is_passable()

    def step_cost(self, row: int, col: int) -> float:
        cell = self.cells[row, col]
        if not cell.is_passable():
            return float("inf")
        # Applica il moltiplicatore del vento se la cella e' WIND
        wind_mult = WIND_MULTIPLIER if cell.cell_type == CellType.WIND else 1.0
        return BASE_COST[cell.cell_type] * wind_mult

    def mark_delivered(self, row: int, col: int) -> bool:
        """Segna una consegna come completata. Restituisce True se era pendente."""
        cell = self.cells[row, col]
        if cell.has_pending_delivery():
            cell.delivered = True
            return True
        return False

    def reset_deliveries(self) -> None:
        """Reimposta tutte le consegne come non effettuate (usato tra un run e l'altro)."""
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[r, c].delivered = False

    def pending_deliveries(self) -> list[tuple[int, int]]:
        """Restituisce le posizioni delle consegne ancora da effettuare."""
        return [p for p in self.delivery_positions
                if self.cells[p[0], p[1]].has_pending_delivery()]

    def manhattan(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """
        Distanza di Manhattan tra due celle.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _find_unique(self, cell_type: CellType, required: bool = True) -> Optional[tuple[int, int]]:
        """Trova l'unica cella di un certo tipo (lancia errore se ce ne sono piu' di una)."""
        positions = self._find_all(cell_type)
        if not positions:
            if required:
                raise ValueError(f"Nessuna cella {cell_type.name} trovata.")
            return None
        if len(positions) > 1:
            raise ValueError(f"Trovate {len(positions)} celle {cell_type.name} — deve essere unica.")
        return positions[0]

    def _find_all(self, *cell_types: CellType) -> list[tuple[int, int]]:
        """Restituisce tutte le posizioni di celle che corrispondono ai tipi indicati."""
        return [(r, c)
                for r in range(self.rows)
                for c in range(self.cols)
                if self.cells[r, c].cell_type in cell_types]

    CELL_SYMBOLS = {
        CellType.EMPTY:    ".",
        CellType.START:    "S",
        CellType.BLOCKED:  "X",
        CellType.DELIVERY: "D",
        CellType.WIND:     "W",
    }

    def ascii_render(self, drone_pos: Optional[tuple[int, int]] = None) -> str:
        """
        Stampa la griglia in ASCII — utile per debug e verifica visiva.
        """
        lines = []
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if drone_pos and (r, c) == drone_pos:
                    row_str += "@"
                else:
                    cell = self.cells[r, c]
                    sym  = self.CELL_SYMBOLS[cell.cell_type]
                    row_str += sym.lower() if cell.delivered else sym
            lines.append(row_str)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Grid({self.rows}x{self.cols}, start={self.start_pos})"
