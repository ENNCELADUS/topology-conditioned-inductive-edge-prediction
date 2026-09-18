# src/

Implementation of the benchmark/evaluation stack, the shared endpoint-only V3.1
backbone, and the four topology families on top of it (topology prompt, KD from
the Full-Ego oracle, structural objectives, motif-graph prompt). Binding
contracts: [`CLAUDE.md`](../CLAUDE.md) and the
[experiment protocol](../docs/03-experiments.md).

- `data/` — `artifacts.py` (verified benchmark loader), `partition.py`
  (`build_g_struct` is the only legal structural graph), `val_region.py` (V_val),
  `packed_features.py` / `features.py` (BF16 pack + F0 cache), `pairs.py`
  (batching), `struct_coords.py` (the 34 fixed-semantics coordinates),
  `motif_template.py` (the fixed 26-slot template), `grounding.py`
  (arm-specific support only).
- `model/egostitch/` — generator, encoder and classifier slots behind
  `registry.py`, composed by `composite.py`, talking through `graph.py`
  dataclasses. Each slot is chosen by its config section's `name`, so a new
  component is one registry entry, never a `composite.py` change. The pairwise
  backbone lives here as `classifier/b0_v31.py`; the prompt interfaces are
  `classifier/{coord_gen,motif_prompt}.py`. No encoder may read
  `ImaginedGraph.aux` — it is generator-private.
- `distill/` — `config.py` (legal KD weight patterns per arm), `losses.py`,
  `motif_losses.py`, `artifacts.py` (strict bank loaders),
  `teacher_targets.py` / `context_sampler.py` (bank dumping).
- `eval/` — edge and assembled-graph metrics, calibration, checkpoint selection,
  and `test_protocol.py` (the held-out protocol).
- `experiments/` — KD audits, the Optuna drivers (`struct_hpo.py`,
  `kd_*_hpo.py`, `topology_prompt_hpo.py`), and the legacy G1–G3 gates.
- `autoresearch/` — the KD-campaign judge, ledger and curves (governed by the
  human-owned `autoresearch/program.md`; do not edit).
- `train_{b0,egostitch,cazi_mbn,l3ppi}.py` — DDP workers; `e2_pipeline.py` runs
  `pack → train → publish → test`; `score_universe.py` and `score_fanout.py`
  produce score-once artifacts and their multi-GPU fan-out.
- `baselines/` — the isolated external reproductions (TUnA / PPITrans via
  `official_ppi.py`, CAZI-MBN, L3-PPI), adapted only to this protocol.
- `vendor/` — official GRIT and Set Transformer code; do not refactor or
  lint-fix it.

Students never see graph structure: only the KD banks under `outputs/distill/`
carry it. Runs that read true structure of the universe they are scored on are
ceiling diagnostics, published as `diagnostic_*` artifacts.
