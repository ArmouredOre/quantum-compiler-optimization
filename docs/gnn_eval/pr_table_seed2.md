## GNN vs rule-based: precision / recall / F1 (held-out test, verified ground truth)

- seed 2, conv `sage`, 50 epochs
- circuits: 11 usable of 11 (skipped: 0); split 7/2/2 (train/val/test)
- rule candidates passing Module C's check: 68 of 68
- thresholds calibrated on val: {'rule': 0.95, 'verified': 0.95}

| System | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Rule-based generator | 1.000 | 1.000 | 1.000 | 16 | 0 | 0 |
| Adjacent-inverse baseline | 1.000 | 0.000 | 0.000 | 0 | 0 | 16 |
| GNN (rule labels, t=0.95) | 0.171 | 0.375 | 0.235 | 6 | 29 | 10 |
| GNN (verified labels, t=0.95) | 0.171 | 0.375 | 0.235 | 6 | 29 | 10 |
