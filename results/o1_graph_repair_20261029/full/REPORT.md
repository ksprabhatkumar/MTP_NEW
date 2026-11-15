# Graph repair validation pilot

Checkpoints selected only by source ideal macro-F1. Values are means over validation subjects; no held-out test evaluated.

| Arm | Ideal accuracy | Self-placement accuracy | Self-placement macro-F1 |
|---|---:|---:|---:|
| none_clean | 94.754% | 93.032% | 93.083% |
| none_robust | 95.114% | 93.907% | 93.995% |
| local_repair | 95.421% | 94.368% | 94.373% |
| graph_repair | 95.114% | 93.904% | 94.020% |

Graph minus robust no-graph self-placement accuracy: -0.003 percentage points.

See metrics.csv for individual folds, learning_curves.csv for every epoch, and jobs/*/* for predictions, class metrics, checkpoints, and graph interventions.