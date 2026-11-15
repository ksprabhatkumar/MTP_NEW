# Receiver-blind predictive repair validation pilot

Checkpoints selected only by source ideal macro-F1. Values are means over validation subjects; no held-out test evaluated.

| Arm | Ideal accuracy | Self-placement accuracy | Self-placement macro-F1 |
|---|---:|---:|---:|
| none_clean | 95.117% | 92.894% | 92.731% |
| none_robust | 96.285% | 94.334% | 94.060% |
| local_repair | 96.392% | 94.287% | 94.159% |
| graph_repair | 96.406% | 94.529% | 94.278% |

Graph minus robust no-graph self-placement accuracy: +0.195 percentage points.

See metrics.csv for individual folds, learning_curves.csv for every epoch, and jobs/*/* for predictions, class metrics, checkpoints, and graph interventions.