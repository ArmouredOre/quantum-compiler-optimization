"""qco — hybrid quantum compiler optimizer (Team 2, Review 1).

Sub-packages map 1-to-1 onto the six architecture stages
(see docs/architecture.md):

    qco.ir       -> Stage 1  Front End (parser + IR)
    qco.graphs   -> Stage 2 + 3  graph construction and partitioning
    qco.modules  -> Stage 4 + 5  optimization modules A/B/C and D
    qco.evaluation -> Stage 6  evaluation & benchmarking engine
    qco.pipeline -> wires Stage 1..6 with the closed-loop feedback edge
"""

__version__ = "0.2.0"
