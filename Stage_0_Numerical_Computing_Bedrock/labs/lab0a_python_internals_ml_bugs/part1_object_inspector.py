"""Lab 0A - Part 1: CPython Object, Memory & Generator Inspector.

Run directly:
    python3 part1_object_inspector.py

Observe real memory addresses (id), object byte sizes (sys.getsizeof),
shallow vs deep copy sharing, and eager list vs lazy generator RAM usage.
"""

import copy
import sys


def banner(title: str) -> None:
    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")


def main() -> None:
    banner("1. CPython PyObject Overhead vs Raw C Types")
    x = 42
    y = 3.14159
    print(f"int   x = {x:<10} | id={hex(id(x))} | size={sys.getsizeof(x)} bytes (C int32_t = 4 bytes!)")
    print(f"float y = {y:<10} | id={hex(id(y))} | size={sys.getsizeof(y)} bytes (C double  = 8 bytes!)")

    banner("2. Small Integer Interning (-5..256) vs Large Integers")
    a_small, b_small = 100, 100
    a_big = int("100000")
    b_big = int("100000")
    print(f"Small ints (100):     a == b is {a_small == b_small}, a is b is {a_small is b_small} (interned!)")
    print(f"Large ints (100000):  a == b is {a_big == b_big}, a is b is {a_big is b_big} (separate heap objects!)")
    print("-> Lesson: NEVER use `is` to compare numbers or strings; reserve `is` for `None`!")

    banner("3. Aliasing vs Shallow Copy vs Deep Copy on a 2D Dataset")
    raw = [[1.0, 2.0], [3.0, 4.0]]
    alias = raw
    shallow = raw.copy()
    deep = copy.deepcopy(raw)

    print(f"raw     outer id={hex(id(raw))} | row0 id={hex(id(raw[0]))}")
    print(f"alias   outer id={hex(id(alias))} | row0 id={hex(id(alias[0]))}  <-- SAME outer & row0!")
    print(f"shallow outer id={hex(id(shallow))} | row0 id={hex(id(shallow[0]))}  <-- NEW outer, SHARED row0!")
    print(f"deep    outer id={hex(id(deep))} | row0 id={hex(id(deep[0]))}  <-- FULLY ISOLATED!")

    banner("4. The [[0.0] * 3] * 3 Weight Matrix Row-Aliasing Trap")
    bad_w = [[0.0] * 3] * 3
    good_w = [[0.0 for _ in range(3)] for _ in range(3)]
    print("bad_w  row IDs:", [hex(id(row)) for row in bad_w], "<- ALL 3 IDENTICAL!")
    print("good_w row IDs:", [hex(id(row)) for row in good_w], "<- 3 DISTINCT LISTS!")

    banner("5. Eager List vs Lazy Generator Memory Footprint (N = 200,000)")
    n = 200_000
    eager_list = [i * 0.5 for i in range(n)]
    lazy_gen = (i * 0.5 for i in range(n))
    print(f"Eager list container size : {sys.getsizeof(eager_list):>10,} bytes (+ ~24 bytes * {n:,} float objects!)")
    print(f"Lazy generator size       : {sys.getsizeof(lazy_gen):>10,} bytes (O(1) constant RAM!)")

    banner("6. Division & Modulo Reality Check (Python vs C/C++)")
    print(f"Python  5 / 2   = {5 / 2}   (always float)")
    print(f"Python -7 // 2  = {-7 // 2}    (floors toward -inf; C++ -7 / 2 == -3!)")
    print(f"Python -7 % 2   = {-7 % 2}     (keeps divisor sign; C++ -7 % 2 == -1!)")


if __name__ == "__main__":
    main()
