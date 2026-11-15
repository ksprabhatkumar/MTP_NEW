# Graph repair validation pilot

Checkpoints selected only by source ideal macro-F1. Values are means over validation subjects; no held-out test evaluated.

| Arm | Ideal accuracy | Self-placement accuracy | Self-placement macro-F1 |
|---|---:|---:|---:|
| none_clean | 90.044% | 82.651% | 78.316% |
| none_robust | 90.292% | 82.166% | 77.809% |
| local_repair | 90.604% | 82.435% | 77.990% |
| graph_repair | 90.292% | 82.274% | 77.913% |

Graph minus robust no-graph self-placement accuracy: +0.108 percentage points.

See metrics.csv for individual folds, learning_curves.csv for every epoch, and jobs/*/* for predictions, class metrics, checkpoints, and graph interventions.