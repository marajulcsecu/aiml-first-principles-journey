# Stage 0: Scientific Python, Numerical Computing & Experiment Bedrock

- **Interactive Mastery Syllabus:** [`stage_0_detailed_plan.html`](./stage_0_detailed_plan.html)
- **Master Roadmap:** [`../master_roadmap.html`](../master_roadmap.html)

## Modules (Self-Paced, Mastery-Gated, Local CPU-First)
1. **Module 0.1 ★ (Active):** Python Complete Foundation & Internals *(C/C++ → Python Lens: Syntax Revision, Object/Reference Model, Aliasing, Generators, OOP & Dunder Methods)*
   - Visual HTML Lecture (English Deep-Dive): [`lessons/module0_1_python_for_ml_engineers.html`](./lessons/module0_1_python_for_ml_engineers.html)
   - Visual HTML Lecture (Pure বাংলা সংস্করণ): [`lessons/module0_1_python_for_ml_engineers_bn.html`](./lessons/module0_1_python_for_ml_engineers_bn.html)
   - Pure Python Lab 0A: [`labs/lab0a_python_internals_ml_bugs/README.md`](./labs/lab0a_python_internals_ml_bugs/README.md)
2. **Module 0.2:** NumPy Core, Vectorization, Broadcasting & Memory Strides *(C-Contiguous Layouts, Views vs. Copies, 100× Speedups)*
3. **Module 0.3:** Floating-Point Reality (`IEEE-754`, `FP32`/`FP16`/`BF16`) & Numerical Stability *(Overflow, Underflow, `log-sum-exp`, Catastrophic Cancellation, `NaN`/`Inf` Forensics)*
4. **Module 0.4:** Exploratory Data Analysis, Visualization & Leakage Hygiene (`Pandas`, `Matplotlib`, Clean Train/Val/Test Discipline)
5. **Module 0.5:** Reproducible Scientific Python Engineering (`venv`, Deterministic Seeds, Profiling, Numerical Unit Tests)
6. **Mastery Gate 0:** Build a zero-dependency vectorized data & numerically stable softmax/cross-entropy mini-toolkit in Python/NumPy, diagnose 4 real silent numerical/data bugs, and deliver a Feynman explanation from memory.

## Directory Layout
- `lessons/` — Interactive Dual-Language Visual HTML Lectures (English + Pure বাংলা)
- `labs/` — Reproducible Pure Python / NumPy Labs + Automated Verifiers
- `writeups/` — Student's own Feynman summaries, derivations & verified proofs
- `tools/` — Clean reusable scientific Python utilities built by the student
