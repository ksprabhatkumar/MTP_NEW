# O1 latent message-correction pilot

Eight-epoch source-only pilot (unless overridden in fingerprint). No held-out placement evaluation.

| Arm | Source ideal F1 | Source self F1 |
|---|---:|---:|
| none | 93.29% | 92.62% |
| local_control | 93.46% | 92.67% |
| message_correction | 93.42% | 92.88% |

Review paired fold effects and correction diagnostics before selecting a full study. This pilot does not establish test superiority or a Mamba/SSM result. CPU staging changes numerical/data-order details from historical GPU-resident runs; use the fresh matched controls.
