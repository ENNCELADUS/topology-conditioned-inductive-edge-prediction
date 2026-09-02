# KD representation-target audit (2026-09-02)

Which teacher vector, if any, is worth distilling into the endpoint-only student. Three row banks from
the same Full-Ego teacher checkpoint `c390709fb7070c62` (PMA(1) bank from its own teacher): `topo` and
`pma1` hold the topology-branch pooled vector $0.5(\text{pooled}_{ab}+\text{pooled}_{ba})$ taken before
fusion; `fused` holds the classifier's pre-head pair feature (AB/BA max after the `pooled_adapter`
injection), dumped with `python -m src.distill.teacher_targets --rep-source fused`. Audit:
`python -m src.experiments.kd_rep_audit` over 86,498 train / 22,804 V_val rows; raw output in
[audit.json](audit.json). Probes are ridge regressions fit on 80% of the training rows and scored on
the held-out 20% (in-distribution) and on V_val (internal edges removed, so structurally shifted).

## Model architecture

![Teacher and student architecture](teacher_architecture.svg)

Grey blocks are shared by teacher and student; blue blocks exist only in the teacher, whose
topology representation $t_{uv}$ enters the fusion step (in the code, a zero-init adapter inside the
pairwise encoder) and yields $f_{uv}$, the only input to the classifier head. The student has no
topology branch, so its $f_{uv}$ is $c_{uv}$. Dashed lines mark where each distillation loss attaches:
KD1 on the soft logits, KD2 listwise on logits over context banks, KD3 and KD4 on $t_{uv}$ against $c_{uv}$.

## Spectrum and logit alignment (train block)

| bank | var on top-1 / top-2 / top-8 axes | participation ratio | corr(top-1, logit) | corr(cos-Gram, \|Δp\|) |
|---|---|---:|---:|---:|
| topo | 0.93 / 0.98 / 1.00 | 1.16 | −0.96 | −0.85 |
| pma1 | 0.93 / 0.97 / 1.00 | 1.16 | 0.88 | −0.77 |
| fused | 0.73 / 0.84 / 0.91 | 1.82 | 0.99 | −0.94 |

## Linear read-out of the row's oracle-graph structure ($R^2$, train holdout)

| input | log CN | log deg sum | log deg diff | Jaccard | log Adamic-Adar | teacher logit |
|---|---:|---:|---:|---:|---:|---:|
| topo vector | 0.99 | 0.99 | 0.97 | 0.99 | 0.99 | 0.95 |
| pma1 vector | 0.99 | 0.99 | 0.97 | 0.99 | 0.99 | 0.87 |
| fused vector | 0.90 | 0.66 | 0.28 | 0.75 | 0.88 | 1.00 |
| content `[f_u+f_v, \|f_u−f_v\|]` | 0.52 | 0.70 | 0.22 | 0.58 | 0.52 | 0.47 |
| content + teacher logit (diagnostic) | 0.85 | 0.70 | 0.26 | 0.71 | 0.82 | 1.00 |

V_val values are within 0.02 of these for every row except the content+logit and fused degree
columns. Content-to-vector linear predictability is 0.40--0.47 overall for all three banks; per
principal component it is 0.15--0.61 for `topo` (negative on V_val for PC4: split shift, not a bug)
and 0.43--0.82 for `fused`.

## Reading

- The topology-branch vector passes the ego-graph descriptors through almost losslessly, yet 93% of its
  variance is one logit-aligned axis. Cosine and Gram losses weight directions by variance, so KD3 and
  KD4 optimized a re-encoding of KD1's target while the structure sat in the invisible tail.
- The fused vector is more logit-aligned and reads structure no better than content plus the logit:
  fusion keeps only what the edge decision needs. Distilling it would be KD1 plus content self-distillation.
- Any representation KD from the topology vector can transfer at most the content-predictable part.
  The linear bound is $R^2$ 0.2--0.7 (degree sum highest, degree difference lowest); the token-level
  student's true ceiling is unknown and must be measured with a nonlinear probe or an auxiliary head
  before a whitened or descriptor-level KD run is justified.
- The PMA(1) rerun of KD3/KD4 and a fused-target arm are both dropped on this evidence.
