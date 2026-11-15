# Receiver-blind predictive repair validation pilot

Checkpoints selected only by source ideal macro-F1. Values are means over validation subjects; no held-out test evaluated.

| Arm | Ideal accuracy | Self-placement accuracy | Self-placement macro-F1 |
|---|---:|---:|---:|
| none_clean | 90.044% | 82.651% | 78.316% |
| none_robust | 90.604% | 83.082% | 77.941% |
| local_repair | 90.915% | 83.890% | 78.593% |
| graph_repair | 90.853% | 83.944% | 78.726% |

Graph minus robust no-graph self-placement accuracy: +0.862 percentage points.

See metrics.csv for individual folds, learning_curves.csv for every epoch, and jobs/*/* for predictions, class metrics, checkpoints, and graph interventions.