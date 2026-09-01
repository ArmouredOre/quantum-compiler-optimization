# Work Distribution — 5 Members

Each member owns one architecture module end‑to‑end (literature deep‑dive,
design, implementation, evaluation). Member 5 additionally owns system
integration and benchmarking.

| Member | Owned module | Key responsibilities | Package |
|--------|--------------|----------------------|---------|
| **Member 1** | Circuit IR, DAG/Hypergraph construction & Graph Partitioning‑Scheduling | Build QASM/Qiskit parser and IR; DAG builder; qubit‑interaction hypergraph builder; multilevel graph‑partitioning block scheduler; coordinate literature‑review compilation | `qco.ir`, `qco.graphs` |
| **Member 2** | Module A — RL‑Based Adaptive Gate Scheduler | Design RL environment (state, action, reward) over the circuit DAG; implement PPO/DDQN agent; train on small circuits and evaluate generalization to larger ones | `qco.modules.rl_scheduler` |
| **Member 3** | Module B — GNN‑Based Gate Cancellation Predictor | Design gate/node feature encoding; implement GraphSAGE/GAT model; generate labeled training data (valid & profitable cancellations); evaluate precision/recall | `qco.modules.gnn_cancellation` |
| **Member 4** | Module C — SAT/SMT‑Based Global Optimizer & Verifier | Formulate SMT equivalence‑checking encoding for proposed rewrites; implement bounded‑window exact re‑optimization; integrate Z3/CVC5 solver calls | `qco.modules.smt_verifier` |
| **Member 5** | Module D — Evolutionary Multi‑Objective Compiler **+** Integration & Evaluation | Implement NSGA‑II search (pymoo/DEAP) over candidate circuits; build the Evaluation & Benchmarking Engine for the five parameters; integrate all modules into one pipeline; prepare benchmark suite, comparison plots, final report/presentation | `qco.modules.evolutionary`, `qco.evaluation`, `qco.pipeline` |
