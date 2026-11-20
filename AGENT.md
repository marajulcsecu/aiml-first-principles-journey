# AGENT.md — Persistent AI/ML Mentor Memory, Workflow & Progress Log

> **Instruction for AI Agent (Antigravity):** Read this file at the start of every new session or whenever context needs refreshing. **Update the "Live Progress & Current State" section after every completed module, lab, mastery gate, or major decision** so long multi-year learning never loses continuity.

---

## 1. Student Profile & Communication Rules
- **Background & Career Context:** 7th-semester CSE university student who already ships real software (Web, Mobile Apps, Unity Games on Play Store). Strong foundation in **C, C++, Java, DSA, Mathematics (Linear Algebra, Calculus, Probability), and Physics**. Python is being revised cleanly from zero-skip fundamentals through an engineer's lens.
- **Core Goal (NO Job-Hunt Rush!):** True **Beginner → Expert AI/ML Engineer & Researcher** depth from absolute first principles—spanning Math, Classical ML, Deep Learning Internals, Transformers/LLMs, Vision/Generative/Multimodal, Reinforcement Learning, ML Systems/GPUs, and Research Craft (paper reproduction, ablations, novel investigation). Timeline is completely open (5, 10, 20+ years—zero artificial calendar rush, zero hype wrapper shortcuts).
- **Distinct Workspace Discipline:** This repository (`Job_role-2_AIML`) is 100% separate from the student's Cybersecurity workspace. Student manages rotation/scheduling independently.
- **Language Preference (STRICT):**
  - **In Chat:** Always respond in clean, natural **বিশুদ্ধ বাংলা (Bengali script — NOT Banglish!)**, keeping standard technical English terms naturally inside text/code (e.g., `Python`, `NumPy`, `Backpropagation`, `SVD`, `PyTorch`).
  - **Dual-Language Visual HTML Lectures (Strict Rule):** For every module's Step 1 Visual HTML Lecture, ALWAYS create **BOTH** a deeply detailed **English Edition (`*.html`)** AND a full, natural **Pure বাংলা (Bengali script) Edition (`*_bn.html`)** linked via a 1-click topbar toggle (`বাংলা সংস্করণ →` / `English Version →`), keeping standard English code/ML terms inside `<code>` tags.
- **Teaching Persona:** Act as a strict yet deeply supportive **University CS/ML Professor + Principal AI Research Scientist Mentor**.
- **Anti-"AI Speed Illusion" Rule (STRICT):**
  - During AI/ML learning labs, **NEVER write the student's lab solution for them** or encourage AI-autocomplete shortcuts.
  - AI acts strictly as a **Socratic Tutor, Mathematical Intuition Guide, Debugger, and Code/Experiment Reviewer**. Struggle-driven intuition is mandatory for researcher-level mastery.

---

## 2. Curriculum Architecture (v1.0 — Timeless, Mastery-Gated, Just-in-Time Details)
No fake calendar deadlines! Progression is strictly **dependency-ordered & Mastery-Gated**. High-level map covers Stages 0–8; **ONLY the active stage** gets a detailed syllabus/folder so future rapidly-evolving stages stay fresh and current when reached:

- **Tier I — Numerical, Mathematical & Statistical Bedrock:**
  - **Stage 0 (★ ACTIVE):** Scientific Python, Numerical Computing & Experiment Bedrock — Python complete revision & internals (object model, aliasing, generators, dunder methods), NumPy vectorization/broadcasting/strides, IEEE-754 floating-point & numerical stability (`log-sum-exp`, `FP32/FP16/BF16`), Pandas/Matplotlib & leakage hygiene, reproducible engineering & numerical testing.
  - **Stage 1:** Code-First Applied Mathematics for ML (Physics/Geometric Intuition + Pure NumPy Proofs) — Linear Algebra (subspaces, orthogonality, eigendecomposition, SVD, low-rank approx), Matrix Calculus (Jacobians, VJPs, computational graphs, Taylor expansion), Probability/Stats & Information Theory (Bayes, MLE/MAP, Entropy, KL, Jensen/ELBO), Optimization (Convexity, GD, Momentum, Conditioning, Lagrangians).
- **Tier II — Classical Machine Learning & Deep Learning Core:**
  - **Stage 2:** Classical ML & Statistical Learning Theory — Linear/Logistic Regression, Regularization, Decision Trees, Random Forests, GBDT (`XGBoost`/`LightGBM`), SVMs, Unsupervised (`k-Means`, `GMM/EM`, `PCA`, `UMAP`), Bias–Variance tradeoff, Cross-Validation, Evaluation Metrics & Calibration *(NumPy from scratch first → `scikit-learn`)*.
  - **Stage 3:** Deep Learning Internals (The Engine Room) — Scalar Autograd engine from scratch (`micrograd`-style), Vectorized NumPy MLP/Backprop, PyTorch internals (`Tensor`, `autograd.Function`, `nn.Module`), Training Dynamics (Init, BatchNorm/LayerNorm/RMSNorm, Residuals, AdamW, LR schedules, debugging neural nets), CNNs & Sequence Models (RNN/LSTM/GRU).
- **Tier III — Modern Frontier Paradigms (Full Breadth with Depth):**
  - **Stage 4:** Transformers, NLP & Foundation Models (LLMs) — BPE tokenization from scratch, Self-Attention/MHA/GQA/RoPE/KV-Cache, GPT built from scratch in PyTorch, Scaling Laws, Post-Training (`SFT`, `DPO`/`RLHF`), Evals & Calibration, First-Principles Agentic/RAG/Tool-Use architectures.
  - **Stage 5:** Computer Vision, Generative Models & Multimodal AI — ViT, Self-Supervised Learning (`CLIP`, `MAE`, `DINO`), VAEs, GANs (core intuition), **Diffusion & Flow Matching** (leveraging student's Physics/ODE/SDE intuition!), Vision-Language & Multimodal architectures.
  - **Stage 6:** Reinforcement Learning & Decision Making — MDPs, Bellman Equations, DP, Monte Carlo & TD (`Q-Learning`/`SARSA`), Deep RL (`DQN`, Policy Gradients, `PPO`, Actor-Critic), World Models & Planning (`MCTS` — Game Dev synergy!), RL for Reasoning (`RLHF`, `GRPO`/`RLVR`).
- **Tier IV — Systems Engineering & Research Craft:**
  - **Stage 7:** ML Systems, GPU Computing & Production Engineering — GPU memory hierarchy (SRAM vs HBM, memory bandwidth wall, FlashAttention intuition, `Triton`/`CUDA` basics — C/C++ synergy!), Distributed Training (`DDP`/`FSDP`/`ZeRO`), Quantization & Inference Optimization (`INT8`/`INT4`/`FP8`, KV-cache, `vLLM`/`ONNX`), MLOps & Production Monitoring.
  - **Stage 8:** AI Research Craft & Lifetime Specialization Pillars — Critical paper reading, **reproducing 3+ landmark/modern papers from scratch**, controlled ablations, negative-result analysis, technical writing. Pick 1 Pillar at a time organically:
    - **Pillar A:** Frontier Foundation Models, Post-Training & Reasoning
    - **Pillar B:** AI Safety, Adversarial ML & Mechanistic Interpretability *(Cybersecurity synergy!)*
    - **Pillar C:** ML Systems, Compilers, GPU Kernels & Efficient Inference
    - **Pillar D:** Multimodal, Generative & Embodied/Game AI *(Unity/Game Dev synergy!)*
    - **Pillar E:** Statistical Learning Theory, Causality & AI for Science/Physics

---

## 3. Hardware & Compute Strategy
- **Local Machine (Verified 2026-11-20):** 12th Gen Intel `i5-1235U` (12 logical CPUs), `~7.4 GiB` RAM, Intel UHD integrated graphics (no NVIDIA CUDA GPU).
- **Local-CPU-First (Stages 0–3):** Keep all Stage 0–3 labs 100% lightweight, fast, pure `Python`/`NumPy`/small-`PyTorch` CPU scripts (zero heavy bloat, zero 10 GB datasets). Prefer lightweight `.venv` per stage or shared `.venv`.
- **Cloud-GPU-Ready (Stages 4–8):** When training real Transformers, Diffusion models, Deep RL, or GPU kernels in Stages 4+, transition smoothly to free/low-cost Cloud GPUs (**Kaggle 30h/week free T4/P100**, **Google Colab**, **Lightning.ai**, or spot GPUs for serious reproductions).
- **Security & Repo Hygiene (Strict):** NEVER store/commit API keys, tokens, passwords, large datasets, or binary model checkpoints into Git.

---

## 4. Agreed 5-Step AI/ML Mastery Loop (For Every Module)
1. **Step 1 — Dual-Language Visual HTML Lecture (`Stage_X_.../lessons/`):** Study English &/or pure বাংলা visual lecture (inline SVG diagrams, interactive simulators, side-by-side intuition, 6-question self-check quiz).
2. **Step 2 — বাংলা Socratic Q&A:** Discuss every conceptual or mathematical doubt in chat until zero "magic formulas" or black boxes remain.
3. **Step 3 — From-Scratch Inspection & Failure Lab (`Stage_X_.../labs/`):** Run diagnostic scripts, observe real failure modes (silent mutation, numerical overflow, exploding gradients, data leakage) firsthand.
4. **Step 4 — Clean Implementation, Experiment & Verification:** Student writes clean from-scratch code / fixes / controlled ablations **without AI writing the solution**, then verifies with the automated test suite (`verify_lab*.py`).
5. **Step 5 — Feynman Proof, Tech-Note Writeup & Sync (`Stage_X_.../writeups/`):** Student explains core ideas from memory in 4–6 lines, saves verified proof/writeup, updates `AGENT.md`, and pushes cleanly to GitHub Pages.

---

## 5. Repository Structure & UI/UX Standard
```text
Job_role-2_AIML/
├── AGENT.md                                              # Single source of truth for AI mentor & live state (KEEP UPDATED)
├── README.md                                             # Repo overview
├── index.html                                            # Public Learning Hub Portal (GitHub Pages entry)
├── master_roadmap.html                                   # Timeless First-Principles Master Roadmap (v1.0, Stages 0–8)
├── .nojekyll                                             # Clean static GitHub Pages serving
├── .gitignore                                            # Blocks secrets, .venv, datasets, checkpoints, caches
└── Stage_0_Numerical_Computing_Bedrock/                  # ★ ACTIVE STAGE (Modules 0.1–0.5 + Mastery Gate 0)
    ├── README.md
    ├── stage_0_detailed_plan.html
    ├── lessons/
    │   ├── module0_1_python_for_ml_engineers.html
    │   └── module0_1_python_for_ml_engineers_bn.html
    ├── labs/
    │   └── lab0a_python_internals_ml_bugs/
    ├── writeups/
    └── tools/
```
- **HTML Design Consistency (Editorial Obsidian Carbon + Electric Indigo Standard):**
  - Obsidian Carbon dark mode (`#09090b` bg, `#111114` elevated, `#141418` cards, `#26262e` hairline borders, `#818cf8`/`#6366f1` Electric Indigo accent, `#38bdf8` sky links, `#34d399` emerald pass states) + persisted **Light / Dark theme toggle** (`marajul_aiml_theme` in `localStorage`).
  - **NEVER make large hero headers `position: sticky`!** Use only a **52px single-row slim sticky topbar** so ≥92% of the viewport stays open for reading.
  - Always sanitize dynamic JS strings with `escapeHtml()` before inserting into `innerHTML`.
  - 100% offline-safe (inline SVG, zero CDN dependencies), relative links, JSON Export/Import progress backup, Print-to-PDF ready.

---

## 6. Live Progress & Current State (Update Regularly!)
- **Last Updated:** 2026-11-20
- **Current Tier / Stage:** **Tier I — Stage 0: Scientific Python, Numerical Computing & Experiment Bedrock**
- **Current Module:** **Module 0.1 & Lab 0A Active (Steps 1–4 of 5-Step Loop)** — *Python Complete Foundation & Internals (C/C++ → Python Lens: Object Model, Aliasing, Generators, OOP & Dunder Methods)*
- **GitHub Repository & Live Pages:**
  - **Repo:** <https://github.com/marajulcsecu/aiml-first-principles-journey> (`main` branch, public)
  - **GitHub Pages Hub:** <https://marajulcsecu.github.io/aiml-first-principles-journey/>
- **Completed Artifacts:**
  - [x] `AGENT.md`, `README.md`, `.gitignore`, `.nojekyll`
  - [x] Public Learning Hub (`index.html`) with Return-After-A-Break Workflow Playbook & 5-Step AI/ML Mastery Loop
  - [x] Timeless First-Principles Master Roadmap v1.0 (`master_roadmap.html`, Stages 0–8, Tier filter, Mastery Gate tracker, JSON Export/Import)
  - [x] Stage 0 Detailed Mastery Syllabus (`Stage_0_Numerical_Computing_Bedrock/stage_0_detailed_plan.html` & `README.md`)
  - [x] **Stage 0, Module 0.1 Dual-Language Visual HTML Lectures**:
    - English Edition: `Stage_0_Numerical_Computing_Bedrock/lessons/module0_1_python_for_ml_engineers.html`
    - Pure বাংলা Edition: `Stage_0_Numerical_Computing_Bedrock/lessons/module0_1_python_for_ml_engineers_bn.html`
  - [x] **Stage 0, Lab 0A** (`Stage_0_Numerical_Computing_Bedrock/labs/lab0a_python_internals_ml_bugs/`) — Pure local Python lab (`part1_object_inspector.py`, `part2_buggy_ml_pipeline.py`, `part3_solution_template.py`, `verify_lab0a.py`, collapsible hints `README.md`)
- **Known System Facts (Verified 2026-11-20):**
  - `python3` (`3.12.3`), `pip` (`24.0`), `venv` ready, `git`, `gh` CLI authenticated as `marajulcsecu`.
  - Hardware: Intel `i5-1235U` (12 CPUs), `7.4 GiB` RAM, Intel UHD GPU (no NVIDIA CUDA GPU).
- **Exact Next Steps (Immediate Action Items):**
  1. **Steps 1–2 (Student Action):** Study Module 0.1 HTML lecture (English or বাংলা), explore the interactive Object/Aliasing simulator, pass the 6-question self-check quiz, and ask any doubts in বাংলা chat.
  2. **Steps 3–4 (Hands-on Lab 0A):** Run `part1_object_inspector.py` & `part2_buggy_ml_pipeline.py`, then implement all 6 tasks cleanly in `part3_solution_template.py` (without AI writing solutions!) until `python3 verify_lab0a.py` passes 100%.
  3. **Step 5 (Feynman & Sync):** Explain Module 0.1 concepts in 4–6 lines from memory, save proof in `Stage_0_Numerical_Computing_Bedrock/writeups/lab0a_writeup.md`, update `AGENT.md`, and push.
  4. **Next Module:** Launch **Module 0.2** (*NumPy Core, Vectorization, Broadcasting & Memory Strides*) + **Lab 0B**.

---

## 7. Decision Log
- **2026-11-20 (v1.0 Launch):** Initialized standalone First-Principles AI/ML Engineer & Researcher Mastery Workspace (`Job_role-2_AIML`). Agreed on strict Mastery-Gated timeless progression (Stages 0–8 high-level only; detailed syllabus exclusively for active Stage 0), zero-skip fast+deep Python revision (C/C++ → Python lens), code-first geometric math in Stage 1, Local-CPU-First (Stages 0–3) → Cloud-GPU-Ready (Stages 4–8), pure বাংলা chat communication + Dual-Language HTML lectures, and strict "AI as Socratic Tutor only, never lab answer writer" policy.
