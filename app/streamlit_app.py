"""Streamlit demo GUI for the hybrid quantum compiler optimizer (#60).

Run locally:   streamlit run app/streamlit_app.py
Deploy:        Streamlit Community Cloud, main file ``app/streamlit_app.py``
               (dependencies come from ``app/requirements.txt``).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works without `pip install -e .` (Streamlit Cloud)

from qco.evaluation.baselines import cx_cost  # noqa: E402
from qco.evaluation.calibration import get_calibration, list_calibrations  # noqa: E402
from qco.evaluation.metrics import estimated_fidelity, execution_time  # noqa: E402
from qco.ir.intermediate_representation import IntermediateRepresentation  # noqa: E402
from qco.ir.parser import from_qasm2  # noqa: E402
from qco.pipeline import HybridOptimizer, PipelineConfig, PipelineResult  # noqa: E402

SUITE = ROOT / "benchmarks" / "circuits"
DRAW_MAX_QUBITS, DRAW_MAX_GATES = 12, 120
SLOW_QUBITS = 16

st.set_page_config(page_title="QCO - Quantum Compiler Optimizer", page_icon="⚛️", layout="wide")


# -- helpers ------------------------------------------------------------------

def stats(ir: IntermediateRepresentation, calibration: str) -> dict[str, float]:
    return {
        "Gates": ir.gate_count(),
        "2-qubit gates": ir.two_qubit_gate_count(),
        "CX-cost": cx_cost(ir),
        "Depth": ir.depth(),
        "Est. fidelity": estimated_fidelity(ir, calibration).value,
        "Exec. time (ns)": execution_time(ir, calibration).value,
    }


def draw(ir: IntermediateRepresentation) -> str | None:
    """ASCII circuit diagram via Qiskit, if installed and the circuit is small."""
    if ir.num_qubits > DRAW_MAX_QUBITS or ir.gate_count() > DRAW_MAX_GATES:
        return None
    try:
        from qiskit import QuantumCircuit

        # Built gate by gate: qiskit.qasm2.loads segfaults on Streamlit's script thread.
        qc = QuantumCircuit(ir.num_qubits)
        for g in ir.gates:
            getattr(qc, g.name)(*g.params, *g.qubits)
        return str(qc.draw("text", fold=-1))
    except Exception:  # noqa: BLE001 - diagram is cosmetic
        return None


def best_point(result: PipelineResult):
    return min(result.pareto_front, key=lambda p: p.objective.as_tuple())


def run_baselines(circuit: IntermediateRepresentation, calibration: str) -> pd.DataFrame:
    """Installed baseline compilers, run in a subprocess (see app/run_baselines.py)."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("run_baselines.py")), calibration],
        input=circuit.to_qasm2(), capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else
                           f"baseline process exited with code {proc.returncode}")
    return pd.DataFrame(json.loads(proc.stdout))


# -- sidebar: input + configuration ----------------------------------------------

st.sidebar.title("⚛️ QCO")
st.sidebar.caption("RL scheduling · GNN cancellation · SMT verification · NSGA-II")

source = st.sidebar.radio("Circuit source", ["Benchmark suite", "Upload .qasm", "Paste QASM"])
qasm_text, circuit_name = "", "circuit"
if source == "Benchmark suite":
    files = sorted(p.name for p in SUITE.glob("*.qasm"))
    choice = st.sidebar.selectbox("Benchmark", files, index=files.index("qft_n5.qasm") if "qft_n5.qasm" in files else 0)
    qasm_text, circuit_name = (SUITE / choice).read_text(), choice.removesuffix(".qasm")
elif source == "Upload .qasm":
    up = st.sidebar.file_uploader("OpenQASM 2.0 file", type=["qasm"])
    if up is not None:
        qasm_text, circuit_name = up.getvalue().decode(), Path(up.name).stem
else:
    qasm_text = st.sidebar.text_area(
        "OpenQASM 2.0", height=200,
        value='OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nh q[0];\nh q[0];\ncx q[0],q[1];\n'
              'rz(0.3) q[2];\nrz(0.4) q[2];\ncx q[1],q[2];\nx q[1];\nx q[1];\n')
    circuit_name = "pasted"

st.sidebar.subheader("Pipeline settings")
calibration = st.sidebar.selectbox(
    "Device calibration", list_calibrations(), help="Gate times and error rates used for fidelity, "
    "execution time and the RL reward.")
st.sidebar.caption(get_calibration(calibration).description)
budget = st.sidebar.slider("Re-synthesis solver budget (s)", 0.0, 30.0, 5.0, 0.5)
iterations = st.sidebar.slider("Closed-loop feedback iterations", 1, 5, 1)
window_ranking = st.sidebar.selectbox(
    "Re-synthesis window ranking (N1)", ["gnn", "rule", "random", "exhaustive"])
with st.sidebar.expander("Modules (ablations)"):
    use_rl = st.checkbox("A: RL scheduler", True)
    use_gnn = st.checkbox("B: GNN cancellation", True)
    resynth = st.checkbox("C: bounded re-synthesis", True)
    verify = st.checkbox("C: verify every rewrite", True)
    use_ea = st.checkbox("D: NSGA-II search", True)
compare = st.sidebar.checkbox("Compare with installed baselines (Qiskit O0-O3, TKET, ...)", True)


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


st.sidebar.caption(
    f"Backends: GNN {'model' if _has('torch_geometric') else 'rule-based fallback (no torch)'} · "
    f"SMT {'Z3' if _has('z3') else 'numeric only (no z3)'} · "
    f"NSGA-II {'pymoo' if _has('pymoo') else 'seed-only fallback'} · RL greedy stand-in unless a model is loaded")

config = PipelineConfig(
    calibration=calibration, solver_budget_s=budget, feedback_iterations=iterations,
    window_ranking=window_ranking, use_rl=use_rl, use_gnn=use_gnn, use_ea=use_ea,
    enable_resynthesis=resynth, verify_rewrites=verify,
)

# -- main: input circuit ---------------------------------------------------------

st.title("Hybrid Quantum Compiler Optimizer")
if not qasm_text.strip():
    st.info("Pick a benchmark, upload a `.qasm` file or paste OpenQASM 2.0 in the sidebar.")
    st.stop()

try:
    circuit = from_qasm2(qasm_text)
    circuit.name = circuit_name
except Exception as exc:  # noqa: BLE001 - surface parser errors to the user
    st.error(f"Could not parse the circuit: {exc}")
    st.stop()

before = stats(circuit, calibration)
st.markdown(f"**{circuit_name}** · {circuit.num_qubits} qubits · {before['Gates']} gates · depth {before['Depth']}")
if circuit.num_qubits >= SLOW_QUBITS:
    st.warning(f"{circuit.num_qubits}-qubit circuit: optimization can take a few minutes.")

run_key = (qasm_text, repr(asdict(config)), compare)
if st.button("Optimize", type="primary"):
    with st.spinner("Running Stages 1-6 ..."):
        t0 = time.perf_counter()
        result = HybridOptimizer(config).run(circuit)
        runtime = time.perf_counter() - t0
        baselines = None
        if compare:
            try:
                baselines = run_baselines(circuit, calibration)
            except Exception as exc:  # noqa: BLE001 - baselines are optional
                st.warning(f"Baseline comparison failed: {exc}")
    st.session_state["run"] = (run_key, result, runtime, baselines)

if "run" not in st.session_state or st.session_state["run"][0] != run_key:
    diagram = draw(circuit)
    if diagram:
        st.code(diagram, language=None)
    st.caption("Press **Optimize** to run the pipeline with the current settings.")
    st.stop()

_, result, runtime, baselines = st.session_state["run"]
best = best_point(result)
after = stats(best.circuit, calibration)

# -- results ---------------------------------------------------------------------

st.subheader("Result (best Pareto point)")
cols = st.columns(5)
for col, key, better_lower in [(cols[0], "Gates", True), (cols[1], "CX-cost", True), (cols[2], "Depth", True),
                               (cols[3], "Est. fidelity", False), (cols[4], "Exec. time (ns)", True)]:
    fmt = "{:.4f}" if key == "Est. fidelity" else "{:,.0f}"
    col.metric(key, fmt.format(after[key]), fmt.format(after[key] - before[key]),
               delta_color="inverse" if better_lower else "normal")

v1, v2, v3, v4, v5 = st.columns(5)
v1.metric("Verified", "yes" if result.verified else "no")
v2.metric("Blocks", result.blocks)
v3.metric("Reverted rewrites", result.reverts)
v4.metric("Unverified blocks", result.unverified_blocks)
v5.metric("Optimizer runtime", f"{runtime:.2f} s", f"solver {result.solver_time_s:.2f} s", delta_color="off")
if not result.verified:
    st.error("The stitched circuit failed verification; the pipeline kept the original circuit.")

tab_pareto, tab_circuit, tab_compare, tab_log = st.tabs(
    ["Pareto front", "Optimized circuit", "Baselines", "Verification log"])

with tab_pareto:
    front = pd.DataFrame([
        {"Point": i, "Gates": p.objective.gate_count, "Depth": p.objective.depth,
         "Surrogate fidelity": 1 - p.objective.infidelity}
        for i, p in enumerate(result.pareto_front)
    ])
    orig = pd.DataFrame([{"Point": "input", "Gates": before["Gates"], "Depth": before["Depth"]}])
    chart = (
        alt.Chart(front).mark_circle(size=140).encode(
            x=alt.X("Gates:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="d")),
            y=alt.Y("Depth:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="d")),
            color=alt.Color("Surrogate fidelity:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["Point", "Gates", "Depth", alt.Tooltip("Surrogate fidelity:Q", format=".5f")])
        + alt.Chart(orig).mark_point(shape="cross", size=200, color="red").encode(
            x="Gates:Q", y="Depth:Q", tooltip=["Point", "Gates", "Depth"])
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Circles: Pareto-optimal variants (all equivalent to the input). Red cross: input circuit. "
               "NSGA-II ranks points with its structure-aware fidelity *surrogate* (idle decay, crosstalk), "
               "so it is lower than the per-gate *Est. fidelity* shown above.")
    st.dataframe(front, hide_index=True, width="stretch")

with tab_circuit:
    idx = st.selectbox("Pareto point", list(range(len(result.pareto_front))),
                       index=result.pareto_front.index(best),
                       format_func=lambda i: f"#{i}: {result.pareto_front[i].objective.gate_count:.0f} gates, "
                                             f"depth {result.pareto_front[i].objective.depth:.0f}")
    chosen = result.pareto_front[idx].circuit
    st.dataframe(pd.DataFrame([{"": "input", **before}, {"": f"point #{idx}", **stats(chosen, calibration)}]),
                 hide_index=True, width="stretch")
    left, right = st.columns(2)
    for col, title, ir in [(left, "Input", circuit), (right, f"Optimized (point #{idx})", chosen)]:
        col.markdown(f"**{title}**")
        diagram = draw(ir)
        col.code(diagram if diagram else ir.to_qasm2(), language=None if diagram else "text")
    st.download_button("Download optimized QASM", chosen.to_qasm2(),
                       file_name=f"{circuit_name}_opt{idx}.qasm", mime="text/plain")

with tab_compare:
    if baselines is None:
        st.info("Enable *Compare with installed baselines* in the sidebar and re-run.")
    elif baselines.empty:
        st.info("No baseline compilers installed (e.g. `pip install qiskit`).")
    else:
        qco_row = pd.DataFrame([{"Optimizer": "qco (this project)", **after, "Runtime (s)": runtime, "Note": ""},
                                {"Optimizer": "input", **before, "Note": ""}])
        st.dataframe(pd.concat([qco_row, baselines], ignore_index=True), hide_index=True,
                     width="stretch")
        st.caption("Tools emit different gate sets; compare **CX-cost** (multi-qubit gates in CX-equivalents).")

with tab_log:
    if result.reward_history:
        st.line_chart(pd.DataFrame({"Reward": result.reward_history}), height=200)
        st.caption("Closed-loop reward per feedback iteration (Stage 6 -> Module A).")
    if not result.revert_events:
        st.success("No rewrite was rejected by the equivalence checker.")
    for ev in result.revert_events:
        with st.expander(f"Block {ev.block_index} · {ev.stage}: {ev.reason[:80]}"):
            st.write(ev.reason)
            if ev.counterexample:
                st.json({k: str(v) for k, v in ev.counterexample.items()})
