"""Lab 0A - Part 3: Your Clean Implementation Template (6 Tasks).

RULES:
  1. Write every implementation yourself (NO AI autopilot!).
  2. Standard library Python ONLY (zero third-party dependencies).
  3. Run `python3 verify_lab0a.py` to test your work.
"""

from collections.abc import Iterator
from typing import Any


# ---------------------------------------------------------------------------
# TASK 1: Fix Mutable Default Argument Pollution
# ---------------------------------------------------------------------------
def record_metric_safe(loss: float, history: list[float] | None = None) -> list[float]:
    """Append `loss` to `history` (creating a fresh list if `history` is None) and return it."""
    raise NotImplementedError("Task 1: Implement record_metric_safe")


# ---------------------------------------------------------------------------
# TASK 2: Non-Mutating 2D Row Normalization
# ---------------------------------------------------------------------------
def normalize_features_safe(dataset: list[list[float]]) -> list[list[float]]:
    """Return a NEW 2D list where each row is divided by max(abs(v) for v in row).

    Requirements:
      - MUST NOT mutate `dataset` or any of its inner row lists.
      - If a row is empty, return an empty list [] for that row.
      - If all values in a row are 0.0 (max_abs == 0.0), keep zeros [0.0, ...] safely.
    """
    raise NotImplementedError("Task 2: Implement normalize_features_safe")


# ---------------------------------------------------------------------------
# TASK 3: Independent Zero Matrix Initialization
# ---------------------------------------------------------------------------
def zeros_matrix(rows: int, cols: int) -> list[list[float]]:
    """Return a `rows x cols` 2D list filled with 0.0 where every row is a distinct list.

    Raise ValueError if `rows < 0` or `cols < 0`.
    """
    raise NotImplementedError("Task 3: Implement zeros_matrix")


# ---------------------------------------------------------------------------
# TASK 4: Closure-Safe Learning Rate Scaler Factory
# ---------------------------------------------------------------------------
def make_lr_schedulers(rates: list[float]) -> list[Any]:
    """Return a list of callables `[f_0, f_1, ...]` where `f_i(grad) == grad * rates[i]`."""
    raise NotImplementedError("Task 4: Implement make_lr_schedulers")


# ---------------------------------------------------------------------------
# TASK 5: PyTorch-Style MiniDataset + Lazy Generator Batcher
# ---------------------------------------------------------------------------
class MiniDataset:
    """Minimal PyTorch-style dataset backed by parallel `features` and `labels` sequences.

    Requirements:
      - `__init__(self, features, labels)`: validate `len(features) == len(labels)`
        (raise ValueError if mismatched). Store independent shallow/deep copies or tuples
        cleanly so length & indexing work reliably.
      - `__len__(self) -> int`: return number of samples.
      - `__getitem__(self, idx)`:
          * If `idx` is an `int` (including negative indices like `-1`), return `(features[idx], labels[idx])`.
          * If `idx` is a `slice`, return a `list` of `(feature, label)` pairs for that slice.
    """

    def __init__(self, features: list[list[float]], labels: list[int]) -> None:
        raise NotImplementedError("Task 5a: Implement MiniDataset.__init__")

    def __len__(self) -> int:
        raise NotImplementedError("Task 5b: Implement MiniDataset.__len__")

    def __getitem__(self, idx: int | slice) -> Any:
        raise NotImplementedError("Task 5c: Implement MiniDataset.__getitem__")


def batch_generator(dataset: MiniDataset, batch_size: int) -> Iterator[list[tuple[list[float], int]]]:
    """Lazily yield consecutive mini-batches of size `batch_size` (last batch may be smaller).

    Requirements:
      - MUST be a real Python generator (use `yield`), NOT returning a pre-built list!
      - Raise `ValueError` if `batch_size <= 0`.
      - If `len(dataset) == 0`, yield nothing cleanly.
    """
    raise NotImplementedError("Task 5d: Implement batch_generator using yield")


# ---------------------------------------------------------------------------
# TASK 6: Autograd-Preview MiniScalar with Operator Overloading & Reflected Ops
# ---------------------------------------------------------------------------
class MiniScalar:
    """A minimal scalar wrapper previewing how Stage 3 Autograd builds expressions.

    Requirements:
      - `__init__(self, data: float)`: store `self.data = float(data)`.
      - `__repr__(self) -> str`: return `f"MiniScalar({self.data})"`.
      - `__add__(self, other)` and `__radd__(self, other)`:
          Support both `MiniScalar + MiniScalar`, `MiniScalar + float/int`, and `float/int + MiniScalar`.
          Return a NEW `MiniScalar`.
      - `__mul__(self, other)` and `__rmul__(self, other)`:
          Support both `MiniScalar * MiniScalar`, `MiniScalar * float/int`, and `float/int * MiniScalar`.
          Return a NEW `MiniScalar`.
      - `__call__(self, x: float) -> float`:
          Behave like a 1-parameter scaling layer returning `self.data * float(x)`.
    """

    def __init__(self, data: float) -> None:
        raise NotImplementedError("Task 6a: Implement MiniScalar.__init__")

    def __repr__(self) -> str:
        raise NotImplementedError("Task 6b: Implement MiniScalar.__repr__")

    def __add__(self, other: "MiniScalar | int | float") -> "MiniScalar":
        raise NotImplementedError("Task 6c: Implement MiniScalar.__add__")

    def __radd__(self, other: "MiniScalar | int | float") -> "MiniScalar":
        raise NotImplementedError("Task 6d: Implement MiniScalar.__radd__")

    def __mul__(self, other: "MiniScalar | int | float") -> "MiniScalar":
        raise NotImplementedError("Task 6e: Implement MiniScalar.__mul__")

    def __rmul__(self, other: "MiniScalar | int | float") -> "MiniScalar":
        raise NotImplementedError("Task 6f: Implement MiniScalar.__rmul__")

    def __call__(self, x: float) -> float:
        raise NotImplementedError("Task 6g: Implement MiniScalar.__call__")
