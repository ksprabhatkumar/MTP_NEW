# Source-only message audit (T03)

No dataset audit or held-out displacement evaluation was repeated.

| Excluded subject | Arm | Validation scenario | Normal F1 % | Zero message F1 % | Window roll F1 % | Zero correction F1 % |
|---|---|---|---:|---:|---:|---:|
| 2 | local_control | val_ideal | 92.3839 | 91.7366 | 92.6460 | 90.3245 |
| 2 | local_control | val_self | 90.1484 | 89.4871 | 89.9517 | 87.4292 |
| 2 | message_correction | val_ideal | 92.2730 | 92.0039 | 92.3668 | 90.3428 |
| 2 | message_correction | val_self | 90.0720 | 90.1780 | 90.0853 | 88.3655 |
| 5 | local_control | val_ideal | 94.5386 | 94.6599 | 94.1695 | 93.3642 |
| 5 | local_control | val_self | 95.2016 | 95.3964 | 94.9561 | 96.0071 |
| 5 | message_correction | val_ideal | 94.5618 | 94.3036 | 94.5751 | 92.9799 |
| 5 | message_correction | val_self | 95.6853 | 95.7019 | 95.7594 | 96.1148 |

Interpretation: zeroing the old graph messages slightly improves both source-self folds; removing the entire correction has mixed, larger effects. The local input to the original correction can account for much of its behavior. This motivates testing a neighbor-only value path. It does not establish a displacement gain or prove all neighbor information is harmful.

Shuffle limitation: whole-window rolling stays inside deterministic evaluation batches. Adjacent windows can overlap or share activities, so this is a weak donor intervention. Do not infer full message independence from its small effect. Zero-message ablation is the stronger evidence here. A future graph mechanism claim should add global distant-window donors and static/uniform adjacency controls before final interpretation.
