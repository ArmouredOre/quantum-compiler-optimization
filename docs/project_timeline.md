# Project Initiation Plan & Timeline

Three review milestones. Review 1 (this report) covers problem identification,
literature survey, architecture design, and task allocation.

| Phase | Duration (indicative) | Deliverables | Checkpoint |
|-------|-----------------------|--------------|-----------|
| **Phase 1 — Problem Study & Literature Survey** | Weeks 1–3 | Finalized problem statement, literature review, identified research gap | — |
| **Phase 2 — Architecture Design & Environment Setup** | Weeks 3–4 | Finalized architecture diagram, tool/library setup, benchmark circuit suite selected | **Review 1** |
| Phase 3 — Module‑wise Prototyping | Weeks 5–9 | Independent working prototypes: IR+partitioner, RL scheduler, GNN predictor, SAT/SMT verifier, EA search | **Review 2** |
| Phase 4 — Integration | Weeks 10–11 | End‑to‑end pipeline combining all four modules with feedback loop | — |
| Phase 5 — Evaluation & Benchmarking | Weeks 12–13 | Comparison against Qiskit O0–O3 on the five parameters, Pareto‑front analysis | — |
| Phase 6 — Final Report & Demonstration | Week 14 | Final report, demo, presentation | **Review 3** |

## Phase 1 + Phase 2 completion (this repository)

* **Phase 1** — `docs/phase1_literature_survey.md`, `docs/research_gap.md`.
* **Phase 2**
  * *Finalized architecture diagram* — `docs/architecture.md` (Mermaid) +
    `docs/architecture-diagram.svg` (vector).
  * *Tool/library setup* — `pyproject.toml`, `requirements.txt`,
    `environment.yml`, `scripts/check_environment.py`, package skeleton under
    `src/qco/` mapping 1‑to‑1 onto the six stages.
  * *Benchmark circuit suite selected* — `benchmarks/suite.yaml` +
    `benchmarks/generate_circuits.py` producing OpenQASM 2.0 files in
    `benchmarks/circuits/`.
