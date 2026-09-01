# Research Gap

* Most existing work optimizes **gate scheduling (RL)**, **gate cancellation
  (GNN)**, **exact re‑synthesis (SAT/SMT)**, or **multi‑objective trade‑offs
  (EA)** *in isolation*. Very few pipelines combine learned heuristics with
  formal correctness verification and Pareto‑based refinement end‑to‑end.
* **RL agents trained on small circuits generalize poorly** without an auxiliary
  deterministic / rule‑based reduction layer — a pure learning approach is not
  sufficient on its own.
* **Fixed compiler pass pipelines (Qiskit O0–O3) are shown to be sub‑optimal**
  because the best pass combination is circuit‑ and hardware‑dependent — an
  argument for a scheduler that adapts per input rather than applying a static
  sequence.
* **GNN‑based predictors are largely used to predict a downstream metric**
  (fidelity, hardware fit, success probability) rather than to directly drive or
  rank cancellation actions inside the optimization loop.
* **SAT/SMT approaches provide optimality guarantees but do not scale**; there is
  limited work using them as a bounded, learning‑guided verification / refinement
  layer rather than as a stand‑alone global synthesizer.
* **Evolutionary multi‑objective methods often rely on full unitary simulation**
  to score fidelity, which does not scale past small qubit counts — motivating
  cheaper structure‑aware surrogates.

## Contribution

Close these gaps by combining all four techniques into a **single closed‑loop
pipeline** with feedback from evaluation back into the RL reward, rather than
treating them as independent alternatives.
