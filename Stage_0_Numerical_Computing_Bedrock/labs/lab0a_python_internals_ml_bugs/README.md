# Lab 0A — Python Internals, Silent ML Bugs & PyTorch Dunder Preview

> **Strict Anti-Autopilot Rule:** Read the code, trace memory with your own brain, and write every line in `part3_solution_template.py` **yourself**—do NOT ask AI to write the solution for you! Ask your mentor for hints or code review *after* you try.

## Objective
Move beyond "knowing Python syntax" to mastering the **CPython object/reference model**, **memory-safe lazy generators**, and the **dunder methods** that power `PyTorch` (`Dataset`, `nn.Module`) and `Autograd`.

## Quick Start (Pure Python 3.12 — Zero External Packages Needed)
```bash
cd Stage_0_Numerical_Computing_Bedrock/labs/lab0a_python_internals_ml_bugs

# Step 3A: Inspect live CPython memory addresses, refcounts & generator footprint
python3 part1_object_inspector.py

# Step 3B: Witness 4 realistic ML bugs that run with ZERO crashes yet corrupt results
python3 part2_buggy_ml_pipeline.py

# Step 4: Implement all 6 tasks in part3_solution_template.py, then verify:
python3 verify_lab0a.py
```

---

## Mission Checklist
1. **Part 1 (`part1_object_inspector.py`):** Run it and verify every memory address (`hex(id(...))`) matches the mental model from Module 0.1.
2. **Part 2 (`part2_buggy_ml_pipeline.py`):** Read the 4 buggy ML functions and identify the exact root cause of each silent bug before looking at hints.
3. **Part 3 (`part3_solution_template.py`):** Implement all 6 tasks cleanly:
   - **Task 1 (`record_metric_safe`):** Fix mutable default argument cross-run pollution.
   - **Task 2 (`normalize_features_safe`):** Return normalized 2D feature rows *without* mutating the caller's raw dataset.
   - **Task 3 (`zeros_matrix`):** Create an independent `rows × cols` matrix of `0.0` (no shared row references!).
   - **Task 4 (`make_lr_schedulers`):** Return independent callable scalers `[f(x) -> x * r]` without late-binding closure bugs.
   - **Task 5 (`MiniDataset` + `batch_generator`):** Implement PyTorch-style `__len__`, `__getitem__` (supporting negative indices & slices), and a lazy `yield` batch generator ($O(\text{batch\_size})$ memory).
   - **Task 6 (`MiniScalar`):** Implement an Autograd-preview scalar class supporting `__add__`, `__radd__`, `__mul__`, `__rmul__`, `__call__` (scale input by `self.data`), and `__repr__` so expressions like `4.0 * a + b` work cleanly!
4. **Verify:** Run `python3 verify_lab0a.py` until **12/12 tests PASS**.
5. **Step 5 (Feynman Proof):** Save your 4–6 line memory-tested explanation + passing test summary in `../../writeups/lab0a_writeup.md`.

---

## Collapsible Hints (Open ONLY If Stuck After 10+ Minutes!)

<details>
<summary>Hint for Task 1 (Mutable Default Argument)</summary>
Use `history: list[float] | None = None` in the signature, and initialize `if history is None: history = []` inside the function body.
</details>

<details>
<summary>Hint for Task 2 (Safe 2D Normalization Without Mutating Caller)</summary>
Compute each feature's new value into brand-new inner lists (list comprehension or `copy.deepcopy`) rather than dividing `dataset[i][j]` in place. Make sure `max_val == 0.0` is handled gracefully (keep `0.0` instead of ZeroDivisionError if a column/row max is 0, or follow the exact docstring!).
</details>

<details>
<summary>Hint for Task 3 (Independent Zero Matrix)</summary>
Never use `[[0.0] * cols] * rows`. Use a nested list comprehension: `[[0.0 for _ in range(cols)] for _ in range(rows)]`.
</details>

<details>
<summary>Hint for Task 4 (Late-Binding Closure)</summary>
Either bind the loop variable as a default parameter `lambda x, rate=r: x * rate`, or define a small inner factory function `def make_fn(rate): return lambda x: x * rate`.
</details>

<details>
<summary>Hint for Task 5 (MiniDataset &amp; Lazy Generator)</summary>
Inside `MiniDataset.__getitem__(self, idx)`, if `isinstance(idx, slice)`, return `list(zip(self.features[idx], self.labels[idx]))`; otherwise return `(self.features[idx], self.labels[idx])`. In `batch_generator(dataset, batch_size)`, raise `ValueError` if `batch_size <= 0`, then loop `range(0, len(dataset), batch_size)` and `yield` one batch slice at a time.
</details>

<details>
<summary>Hint for Task 6 (MiniScalar &amp; Reflected Operators)</summary>
Helper tip: convert `other` to a float via `other.data if isinstance(other, MiniScalar) else float(other)`. Implement `__radd__(self, other)` by delegating to `self.__add__(other)` and `__rmul__(self, other)` by delegating to `self.__mul__(other)` (since scalar addition and multiplication are commutative!).
</details>
