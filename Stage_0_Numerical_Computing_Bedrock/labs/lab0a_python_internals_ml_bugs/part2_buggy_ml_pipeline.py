"""Lab 0A - Part 2: Four Realistic Silent ML Pipeline Bugs.

Run directly:
    python3 part2_buggy_ml_pipeline.py

Every single function below executes with ZERO exceptions or warnings—yet
silently corrupts ML training, weights, or evaluation metrics!
Study each bug carefully before implementing clean fixes in part3_solution_template.py.
"""


# ---------------------------------------------------------------------------
# BUG 1: Mutable Default Argument Pollution Across Experiments
# ---------------------------------------------------------------------------
def buggy_record_metric(loss: float, history: list[float] = []) -> list[float]:
    history.append(loss)
    return history


# ---------------------------------------------------------------------------
# BUG 2: Shallow Copy Mutation Corrupting Caller's Raw Dataset
# ---------------------------------------------------------------------------
def buggy_normalize_rows(dataset: list[list[float]]) -> list[list[float]]:
    """Intended to return a normalized copy without touching `dataset`."""
    copied = dataset.copy()  # BUG: shallow copy! Inner rows still point to `dataset`!
    for row in copied:
        max_abs = max(abs(v) for v in row) if row else 0.0
        if max_abs > 0.0:
            for j in range(len(row)):
                row[j] = row[j] / max_abs
    return copied


# ---------------------------------------------------------------------------
# BUG 3: Shared Row References in Weight Matrix Initialization
# ---------------------------------------------------------------------------
def buggy_zeros_matrix(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols] * rows


# ---------------------------------------------------------------------------
# BUG 4: Late-Binding Closure in Learning Rate Scaler Factory
# ---------------------------------------------------------------------------
def buggy_make_lr_schedulers(rates: list[float]):
    return [lambda grad: grad * r for r in rates]


def main() -> None:
    print("=== BUG 1: Mutable Default Argument Cross-Run Contamination ===")
    exp1_log = buggy_record_metric(0.92)
    exp2_log = buggy_record_metric(0.35)
    print(f"Expected exp2_log == [0.35], ACTUAL: {exp2_log}  <-- Exp 1 leaked into Exp 2!")
    print(f"Are exp1_log and exp2_log the same object? {exp1_log is exp2_log}")

    print("\n=== BUG 2: Shallow Copy Silently Mutates Raw Dataset ===")
    raw_data = [[10.0, 20.0], [5.0, 25.0]]
    print(f"raw_data BEFORE calling buggy_normalize_rows: {raw_data}")
    normed = buggy_normalize_rows(raw_data)
    print(f"normed   returned                           : {normed}")
    print(f"raw_data AFTER  calling buggy_normalize_rows: {raw_data}  <-- SILENTLY CORRUPTED!")

    print("\n=== BUG 3: Updating One Weight Overwrites Every Row ===")
    W = buggy_zeros_matrix(3, 3)
    W[0][1] = 0.75
    print("After setting ONLY W[0][1] = 0.75:")
    for r_idx, row in enumerate(W):
        print(f"  W[{r_idx}] = {row}")

    print("\n=== BUG 4: Late-Binding Closure Captures Final Loop Value ===")
    rates = [0.1, 0.01, 0.001]
    schedulers = buggy_make_lr_schedulers(rates)
    outputs = [fn(100.0) for fn in schedulers]
    print(f"Expected [10.0, 1.0, 0.1], ACTUAL: {outputs}  <-- ALL used 0.001!")


if __name__ == "__main__":
    main()
