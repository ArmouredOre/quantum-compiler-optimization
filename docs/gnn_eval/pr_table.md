## GNN vs rule-based: precision / recall / F1 (held-out test, verified ground truth)

- seed 0, conv `sage`, 50 epochs
- circuits: 11 usable of 11 (skipped: 0); split 7/2/2 (train/val/test)
- rule candidates passing Module C's check: 68 of 68
- thresholds calibrated on val: {'rule': 0.8, 'verified': 0.8}

| System | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Rule-based generator | 1.000 | 1.000 | 1.000 | 40 | 0 | 0 |
| Adjacent-inverse baseline | 1.000 | 0.000 | 0.000 | 0 | 0 | 40 |
| GNN (rule labels, t=0.80) | 0.068 | 0.600 | 0.122 | 24 | 329 | 16 |
| GNN (verified labels, t=0.80) | 0.068 | 0.600 | 0.122 | 24 | 329 | 16 |
