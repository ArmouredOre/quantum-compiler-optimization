"""Module B — GraphSAGE / GAT model that scores cancellation candidates.

Phase 3 (Member 3): implement the message-passing network, the training loop over
labeled candidates (label = "valid AND reduces gate count without raising depth"),
and precision/recall evaluation. Phase 2 exposes the call surface and falls back
to the rule-based baseline so the pipeline runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from qco.ir.intermediate_representation import IntermediateRepresentation
from qco.modules.gnn_cancellation.features import CancellationCandidate, rule_based_candidates


@dataclass(slots=True)
class GNNConfig:
    conv: str = "sage"          # "sage" | "gat"
    hidden: int = 128
    layers: int = 3
    dropout: float = 0.1
    epochs: int = 100
    lr: float = 1e-3


class GNNCancellationPredictor:
    def __init__(self, config: GNNConfig | None = None):
        self.config = config or GNNConfig()
        self._net = None

    def fit(self, dataset) -> None:  # pragma: no cover - Phase 3
        raise NotImplementedError("GNNCancellationPredictor.fit lands in Phase 3 (Member 3)")

    def predict(self, ir: IntermediateRepresentation, threshold: float = 0.5) -> list[CancellationCandidate]:
        """Ranked cancellation candidates (highest score first).

        Falls back to the deterministic rule-based candidate set until a trained
        network is loaded.
        """
        if self._net is None:
            cands = rule_based_candidates(ir)
        else:  # pragma: no cover - Phase 3
            raise NotImplementedError
        return sorted((c for c in cands if c.score >= threshold), key=lambda c: -c.score)
