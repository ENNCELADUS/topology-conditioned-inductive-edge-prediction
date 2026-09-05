# Topology Project Instructions

## Research contract and sources

- This ICLR 2027 project studies topology-conditioned binary edge prediction. Strict task input is exactly `(x_u, x_v)`; output is a symmetric decision for `edge(u, v)`. Inferred topology is intermediate context, not graph generation. Grounding, retrieval and prototypes are optional arm-specific support, never extra test-time task input or a preselected method.
- `docs/01-project-definition.md` defines the task, `docs/02-methodology.md` records method constraints, and `docs/03-experiments.md` defines evaluation and evidence. `README.md` orients; `hpc/README.md` and `hpc/run.sh` govern execution.
- Use `.codex/skills/hpc-execution/SKILL.md` for training, testing, scoring and merges. HPO experiments uses `.claude/skills/autoresearch/SKILL.md` and its human-owned `autoresearch/program.md`, rather than the generic global autoresearch workflow.

## Execution and experiment boundaries

- Sync working code between machines through Git only: commit, push, then pull on the H20 checkout. Do not rsync/scp/tar working trees.
- GPU work uses the H20 container and `hpc/run.sh`, without a scheduler or general qualification ladder. Ordinary jobs auto-size to visible GPUs; sweep-specific masks and runner details are in the runbook. Direct worker invocation and `--max-steps` are debug-only.
- Normal pipeline stages are pack, train, publish and test; `--skip-test` omits held-out evaluation. EgoStitch is oracle diagnostics; use its diagnostic run kind and artifact names from the runbook.
- General experiments have no added plan, registration or qualification gate. KD campaigns retain the specific setup, frozen keys, judge, ledger and stall checkpoints in `autoresearch/program.md`; generic execution autonomy does not authorize changing that program or inventing a baseline.
- Do not add digest pinning, artifact/text-contract verifiers or eligibility/promotion ceremony. Record provenance; operators remain responsible for matching artifacts to the split. Non-finite state, DDP disagreement, data-boundary violations and I/O failures remain fail-closed; quality telemetry such as slot collapse, margins and dispersion does not gate a run.
- Completion artifacts depend on run kind: the runbook distinguishes publication `complete.json`, held-out `test_report.json`/`test_complete.json`, diagnostic equivalents, and failed attempts. Publication alone does not establish successful held-out evaluation.

## Local commands

Local work is macOS/CPU; use `.venv/bin/python -m ...` through `rtk proxy` for exact output. Dependencies are managed by `uv sync`.

```bash
rtk proxy .venv/bin/python -m pytest tests/<file>.py -n0
rtk proxy .venv/bin/python -m pytest -m "not slow and not integration"
rtk proxy .venv/bin/python -m ruff check <changed-paths>
rtk proxy .venv/bin/python -m mypy src tests
```

Pytest uses `--dist loadfile` because tests share in-file state; never use `--dist load`. Use `-n0` for debugging. Production entry examples:

```bash
rtk proxy hpc/run.sh train configs/b0_v31_breadth_first.yaml
rtk proxy hpc/run.sh train configs/b1_kd_logit_breadth_first.yaml
```

## Architecture

- `src/data/partition.py:build_g_struct` defines the legal structural graph; `val_region.py` defines `V_val`; `grounding.py` is arm-specific; `packed_features.py`/`features.py` handle BF16 packs and F0 caches.
- `src/model/egostitch/` composes registered generator, encoder and classifier components through `graph.py` dataclasses. Add a component through its registry, not `composite.py`. Encoders must not read generator-private `ImaginedGraph.aux`, which a generator swap invalidates.
- B1 KD arms use `src/train_b0.py` and `src/distill/`; `src/e2_pipeline.py` orchestrates production stages. `src/score_universe.py` and `src/score_fanout.py` score and merge artifacts; `src/eval/test_protocol.py` evaluates a published checkpoint; `src/experiments/` holds diagnostics.
- Runtime keys are model-family-specific. Consult the HPC skill before interpreting token budgets, world size, raw-token pack paths or pipeline pack paths.

## Evaluation and claims

- Report edge metrics alongside all five topology metrics: BFS-macro GS (edge-set Dice/F1, higher), RD (toward 1), and degree/clustering/spectral MMD ratios (lower).
- Freeze the density-first topology threshold on the `V_val` sampled-set pair union and replay it on test sampled sets. Separately freeze the max-F1 classification threshold on `val_cls` for Accuracy/F1/MCC. AUROC/AUPRC use raw logits; ECE/Brier use raw sigmoid probabilities. No logit shift substitutes for calibration. Exact selection and replay rules live in `docs/03-experiments.md`.
- `src/eval/checkpoint_selection.py` selects by mean rank of V_val AUPRC and the five topology metrics. There is no eligibility predicate; quality must be judged from metrics, not assumed from publication. Keep oracle diagnostics separate from deployable results.

## Data and numerical traps

- `train_graph.pkl` (original train positives plus validation positives) is the `V_val` substrate. K=5 dispersed-seed hashed-frontier BFS grows on its loopless giant component to 20% induced loopless edges. All `V_val`-internal pairs are quarantined; cross-boundary pairs train. `V_val` is pair-disjoint, not fully inductive; test nodes are disjoint. Earlier V_hold caches, packs, thresholds and results do not carry over.
- `exclude_nodes` filters only pair lists; featureless nodes can remain in graphs and buckets. Training topology strips self-loops, but classification self-pairs and official GS/RD/MMD descriptor self-loops remain.
- Grounding caches are universe-specific (`train`, `V_val`, test). Topology and classification use the same train positives, without a message/supervision split; edge-stream structural targets must remove the queried partner and decrement its degree.
- `load_scores` alone does not validate precision. For EgoStitch artifacts use `validate_artifact_precision(artifact, label=...)`; directly calling `validate_score_precision` on an `egostitch_e2e` artifact spuriously reports missing arrays.
- In `generator/assemble.py`, promote inputs to FP32 before cost and marginal products; promotion afterward preserves BF16 quantization.
- `allow_cache_subset=True` in `score_universe.py` can gather an F0 superset without a content check. Exact-order checks do not establish subset provenance.
- Packed-feature manifests depend on `index.json` insertion order. Sorting or reserializing invalidates the pack; F0 contains FP32 means computed before BF16 conversion and cannot be reconstructed exactly from shards.
- Legacy G1 density-matched thresholds count non-self rows against a self-loop-stripped reference, while self-pairs still assemble as loops. Changing either convention changes operating points.
