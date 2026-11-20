"""Lab 0A Automated Verifier.

Run:
    python3 verify_lab0a.py
"""

import inspect
import math
import sys
import traceback

import part3_solution_template as sol


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def test_task1_default_isolation() -> None:
    r1 = sol.record_metric_safe(0.9)
    r2 = sol.record_metric_safe(0.4)
    assert r1 == [0.9], f"Expected [0.9], got {r1}"
    assert r2 == [0.4], f"Expected [0.4], got {r2} (mutable default leaked!)"
    assert r1 is not r2, "Separate calls returned the same list object!"


def test_task1_explicit_history() -> None:
    buf = [1.2]
    out = sol.record_metric_safe(0.8, buf)
    assert out == [1.2, 0.8] and out is buf, "Explicit history list should be updated and returned."


def test_task2_non_mutating_normalize() -> None:
    raw = [[10.0, -20.0], [0.0, 0.0], []]
    snapshot = [row[:] for row in raw]
    normed = sol.normalize_features_safe(raw)

    assert raw == snapshot, f"Caller dataset was silently mutated! raw={raw}"
    assert len(normed) == 3
    assert approx(normed[0][0], 0.5) and approx(normed[0][1], -1.0), f"Wrong row 0: {normed[0]}"
    assert normed[1] == [0.0, 0.0], f"Zero row failed: {normed[1]}"
    assert normed[2] == [], f"Empty row failed: {normed[2]}"
    assert normed[0] is not raw[0], "Returned inner row still aliases caller's inner row!"


def test_task3_zeros_matrix_isolation() -> None:
    W = sol.zeros_matrix(3, 4)
    assert len(W) == 3 and all(len(r) == 4 for r in W), f"Wrong dimensions: {W}"
    W[0][2] = 7.5
    assert W[1][2] == 0.0 and W[2][2] == 0.0, f"Row aliasing detected! W={W}"
    assert len({id(r) for r in W}) == 3, "Row IDs are not distinct!"


def test_task3_zeros_matrix_negative_dims() -> None:
    for r, c in [(-1, 3), (3, -2)]:
        try:
            sol.zeros_matrix(r, c)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for zeros_matrix({r}, {c})")


def test_task4_lr_schedulers_closures() -> None:
    rates = [0.5, 0.1, 0.01, 0.001]
    fns = sol.make_lr_schedulers(rates)
    assert len(fns) == 4
    results = [fn(100.0) for fn in fns]
    expected = [50.0, 10.0, 1.0, 0.1]
    for got, exp in zip(results, expected, strict=True):
        assert approx(got, exp), f"Closure capture bug! Expected {expected}, got {results}"


def test_task5_mini_dataset_indexing_and_slicing() -> None:
    feats = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    labels = [0, 1, 0, 1]
    ds = sol.MiniDataset(feats, labels)
    assert len(ds) == 4
    assert ds[0] == ([1.0, 2.0], 0)
    assert ds[-1] == ([7.0, 8.0], 1), "Negative indexing failed!"
    assert ds[1:3] == [([3.0, 4.0], 1), ([5.0, 6.0], 0)], f"Slicing failed: {ds[1:3]}"


def test_task5_mini_dataset_mismatch_error() -> None:
    try:
        sol.MiniDataset([[1.0]], [0, 1])
    except ValueError:
        return
    raise AssertionError("MiniDataset must raise ValueError when len(features) != len(labels)")


def test_task5_lazy_batch_generator() -> None:
    ds = sol.MiniDataset([[float(i)] for i in range(5)], [i % 2 for i in range(5)])
    gen = sol.batch_generator(ds, batch_size=2)
    assert inspect.isgenerator(gen), "batch_generator MUST be a lazy generator (use `yield`)!"
    batches = list(gen)
    assert len(batches) == 3, f"Expected 3 batches (2, 2, 1), got {len(batches)}"
    assert [len(b) for b in batches] == [2, 2, 1]
    assert batches[-1] == [([4.0], 0)]

    # Invalid batch_size check
    try:
        list(sol.batch_generator(ds, batch_size=0))
    except ValueError:
        return
    raise AssertionError("batch_generator must raise ValueError when batch_size <= 0")


def test_task6_mini_scalar_ops() -> None:
    a = sol.MiniScalar(2.0)
    b = sol.MiniScalar(3.5)
    c = a + b
    d = a * b
    assert isinstance(c, sol.MiniScalar) and approx(c.data, 5.5)
    assert isinstance(d, sol.MiniScalar) and approx(d.data, 7.0)
    assert repr(a) == "MiniScalar(2.0)", f"Unexpected repr: {repr(a)}"


def test_task6_mini_scalar_reflected_ops() -> None:
    a = sol.MiniScalar(2.5)
    b = sol.MiniScalar(1.0)
    expr = 4.0 * a + b + 3
    assert isinstance(expr, sol.MiniScalar) and approx(expr.data, 14.0), f"Expected 14.0, got {expr}"
    expr2 = 10 + a * 2
    assert isinstance(expr2, sol.MiniScalar) and approx(expr2.data, 15.0), f"Expected 15.0, got {expr2}"


def test_task6_mini_scalar_callable() -> None:
    w = sol.MiniScalar(3.0)
    out = w(4.5)
    assert isinstance(out, float) and approx(out, 13.5), f"Expected 13.5, got {out}"


TESTS = [
    ("Task 1a: Mutable default isolation", test_task1_default_isolation),
    ("Task 1b: Explicit history list", test_task1_explicit_history),
    ("Task 2 : Safe 2D normalization (zero mutation)", test_task2_non_mutating_normalize),
    ("Task 3a: Zeros matrix row isolation", test_task3_zeros_matrix_isolation),
    ("Task 3b: Zeros matrix negative dim validation", test_task3_zeros_matrix_negative_dims),
    ("Task 4 : Closure-safe LR schedulers", test_task4_lr_schedulers_closures),
    ("Task 5a: MiniDataset len/index/negative/slice", test_task5_mini_dataset_indexing_and_slicing),
    ("Task 5b: MiniDataset length mismatch error", test_task5_mini_dataset_mismatch_error),
    ("Task 5c: Lazy batch_generator (yield)", test_task5_lazy_batch_generator),
    ("Task 6a: MiniScalar __add__/__mul__/__repr__", test_task6_mini_scalar_ops),
    ("Task 6b: MiniScalar reflected __radd__/__rmul__", test_task6_mini_scalar_reflected_ops),
    ("Task 6c: MiniScalar __call__", test_task6_mini_scalar_callable),
]


def main() -> int:
    passed = 0
    print("=" * 68)
    print("  Lab 0A Verification Suite (12 Tests)")
    print("=" * 68)
    for name, fn in TESTS:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except NotImplementedError as exc:
            print(f"  [TODO] {name} -> {exc}")
        except AssertionError as exc:
            print(f"  [FAIL] {name} -> {exc}")
        except Exception as exc:
            print(f"  [ERR ] {name} -> {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)

    print("-" * 68)
    print(f"  Result: {passed}/{len(TESTS)} passed")
    print("=" * 68)
    if passed == len(TESTS):
        print("  ALL LAB 0A TESTS PASSED! Proceed to Step 5 (Feynman writeup).")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
