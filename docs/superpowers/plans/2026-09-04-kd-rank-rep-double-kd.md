# kd_rank_rep Double-KD Arm + Strict Optuna HPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `kd_rank_rep` B1 arm (strict-LLP rank/dist logit KD + per-row cosine representation KD in one run) and an Optuna driver that searches its three loss weights the way the kd_rank strict sweep does.

**Architecture:** `DistillConfig` gains one legal weight pattern `{w_rank, w_dist, w_rep}`; the trainer's representation gates switch from `arm == "kd_rep"` to a `_REP_COS_ARMS` set and the context-stream gates from `arm == "kd_rank"` to `w_rank > 0`, so the existing `KDRowBank` (rep term, shared forward) and `KDContextStream` (rank/dist, KD-only forwards) simply both run. The kd_rank Optuna driver is parametrized by a `SweepSpec`; a new thin driver module supplies the kd_rank_rep space, priors, and bank/margin flags.

**Tech Stack:** Python 3.11, PyTorch, Optuna (constrained multivariate TPE, sqlite storage), PyYAML, pytest (`-n0 --dist loadfile` when debugging), ruff, mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-04-kd-rank-rep-double-kd-design.md`

## Global Constraints

- Every existing arm stays bit-identical; the matched-control invariant (no `distill:` or all-zero weights ⇒ undistilled baseline) is unchanged.
- No backward-compatibility shims, no digest/contract gates; fail closed only on non-finite state, DDP disagreement, data-boundary violations, I/O failures.
- Lint gate before every commit: `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src tests` (line length 100; mypy has 98 pre-existing errors in 14 EgoStitch-era files — add zero new ones).
- Run tests locally with `.venv/bin/python -m pytest <file> -n0`; never `--dist load`.
- Document edits: a replacement edit never increases the line count of a doc.
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Y1edjqsnfdGUKH45Cht61H
  ```
- GPU runs happen only on the H20 container via `hpc/run.sh`; nothing in this plan trains.

## File map

| File | Responsibility |
|---|---|
| `src/distill/config.py` (modify) | legal pattern `{w_rank, w_dist, w_rep}` → arm `kd_rank_rep`; context path required when `w_rank` active |
| `src/eval/early_stopping.py` (modify) | `_VAL_TERMS["kd_rank_rep"]` |
| `src/train_b0.py` (modify) | `_REP_COS_ARMS`; `KDRowBank`/`_evaluate_distributed` rep gates; `KDContextStream` + `main` gate on `w_rank` |
| `src/experiments/kd_rank_strict_hpo.py` (modify) | `SweepSpec`, `_write_trial_config`, defaults that keep kd_rank behaviour |
| `src/experiments/kd_rank_rep_hpo.py` (create) | kd_rank_rep space, priors, `--bank/--margin` flags, entry point |
| `configs/autoresearch/kd_rank_rep.yaml` (create) | sweep base config |
| `tests/distill/test_distill_config.py`, `tests/eval/test_early_stopping.py`, `tests/test_train_b0_kd.py`, `tests/experiments/test_kd_rank_rep_hpo.py` | tests |
| `docs/03-experiments.md`, `hpc/README.md` | §1.4 row, §1.5 HPO sentence, launch line |

---

### Task 1: `DistillConfig` learns the `kd_rank_rep` pattern

**Files:**
- Modify: `src/distill/config.py:1-21` (docstring), `:107-131` (`__post_init__` patterns), `:148-163` (`arm`)
- Test: `tests/distill/test_distill_config.py`

**Interfaces:**
- Produces: `DistillConfig(targets_path=..., context_targets_path=..., w_rank>0, w_dist>0, w_rep>0).arm == "kd_rank_rep"`. Every later task keys on this arm string and on `distill.w_rank > 0`.

- [ ] **Step 1: Write the failing tests** — append after `test_kd_gen_arm_pattern` in `tests/distill/test_distill_config.py`:

```python
def test_kd_rank_rep_arm_pattern() -> None:
    cfg = DistillConfig(
        targets_path="t", context_targets_path="c", w_rank=0.1, w_dist=10.0, w_rep=1.0
    )
    assert cfg.active
    assert cfg.arm == "kd_rank_rep"


def test_kd_rank_rep_requires_all_three_weights_and_the_context_bank() -> None:
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig(targets_path="t", context_targets_path="c", w_rank=1.0, w_rep=1.0)
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig(targets_path="t", context_targets_path="c", w_dist=1.0, w_rep=1.0)
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig(
            targets_path="t", context_targets_path="c", w_rank=1.0, w_dist=1.0, w_logit=1.0
        )
    with pytest.raises(ValueError, match="context_targets_path is required"):
        DistillConfig(targets_path="t", w_rank=1.0, w_dist=1.0, w_rep=1.0)
    with pytest.raises(ValueError, match="targets_path is required"):
        DistillConfig(context_targets_path="c", w_rank=1.0, w_dist=1.0, w_rep=1.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/distill/test_distill_config.py -n0 -k kd_rank_rep -v`
Expected: both FAIL with `ValueError: distill weights must follow exactly one arm group`.

- [ ] **Step 3: Implement** — in `src/distill/config.py`:

In `legal_patterns`, add after `frozenset({"w_rank", "w_dist"}),`:
```python
            frozenset({"w_rank", "w_dist", "w_rep"}),
```
Replace the error message body so it reads:
```python
            raise ValueError(
                "distill weights must follow exactly one arm group -- all zero, only "
                "w_logit (kd_logit), w_rank and w_dist (kd_rank), w_rank, w_dist and w_rep "
                "(kd_rank_rep), only w_gram (kd_gram), only w_rep (kd_rep), only w_gen (kd_gen), "
                f"only w_struct (kd_struct), or only w_white (kd_white); got nonzero weights "
                f"{sorted(nonzero)}"
            )
```
Replace `kd_rank_active = nonzero == frozenset({"w_rank", "w_dist"})` with:
```python
        kd_rank_active = "w_rank" in nonzero
```
In `arm`'s `mapped` dict add after the kd_rank entry:
```python
            frozenset({"w_rank", "w_dist", "w_rep"}): "kd_rank_rep",
```
In the module docstring, replace these three lines:
```
one arm group's weight(s) may be nonzero at a time:
``kd_logit`` (`w_logit`, pointwise soft-target logit KD), ``kd_rank``
(`w_rank` + `w_dist`, anchor ranking/distribution KD), ``kd_gram`` (`w_gram`,
```
with these three (same line count):
```
one arm group's weight(s) may be nonzero at a time: ``kd_logit`` (`w_logit`,
pointwise soft-target logit KD), ``kd_rank`` (`w_rank` + `w_dist`, anchor
ranking/distribution KD), ``kd_rank_rep`` (kd_rank plus `w_rep`), ``kd_gram`` (`w_gram`,
```
In the `context_targets_path` attribute doc replace `Required exactly when the ``kd_rank`` arm is active.` with `Required exactly when ``w_rank`` is active (kd_rank, kd_rank_rep).`

- [ ] **Step 4: Run the config tests**

Run: `.venv/bin/python -m pytest tests/distill/test_distill_config.py -n0 -v`
Expected: all PASS (including the pre-existing `exactly one arm group` rejections).

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src/distill/config.py tests/distill/test_distill_config.py
git add src/distill/config.py tests/distill/test_distill_config.py
git commit -m "feat(distill): legal kd_rank_rep weight pattern (w_rank + w_dist + w_rep)"
```

---

### Task 2: Early stopping counts all three validation terms

**Files:**
- Modify: `src/eval/early_stopping.py:35-45` (`_VAL_TERMS`)
- Test: `tests/eval/test_early_stopping.py`

**Interfaces:**
- Consumes: `DistillConfig.arm == "kd_rank_rep"` from Task 1.
- Produces: `compose_val_total(task, kd, distill)` for a kd_rank_rep config sums `w_rank*val_kd_rank_loss + w_dist*val_kd_dist_loss + w_rep*val_kd_rep_loss`; `val_total_terms` lists the three.

- [ ] **Step 1: Write the failing tests** — inside `class TestComposeValTotal`, after `test_kd_rep_uses_the_one_minus_cosine_term_not_the_cosine`:

```python
    def test_kd_rank_rep_adds_rank_dist_and_rep_terms(self) -> None:
        distill = DistillConfig(
            targets_path="t", context_targets_path="c", w_rank=3.0, w_dist=0.5, w_rep=2.0
        )
        kd = {"val_kd_rank_loss": 0.2, "val_kd_dist_loss": 0.4, "val_kd_rep_loss": 0.1}
        assert compose_val_total(0.25, kd, distill) == pytest.approx(
            0.25 + 3.0 * 0.2 + 0.5 * 0.4 + 2.0 * 0.1
        )

    def test_kd_rank_rep_missing_rep_counterpart_raises(self) -> None:
        distill = DistillConfig(
            targets_path="t", context_targets_path="c", w_rank=1.0, w_dist=1.0, w_rep=1.0
        )
        with pytest.raises(RuntimeError, match="val_kd_rep_loss"):
            compose_val_total(0.25, {"val_kd_rank_loss": 0.2, "val_kd_dist_loss": 0.4}, distill)
```

And in the `val_total_terms` test class (the one containing `test_undistilled_names_only_the_task_term`):

```python
    def test_kd_rank_rep_names_three_weighted_terms(self) -> None:
        distill = DistillConfig(
            targets_path="t", context_targets_path="c", w_rank=0.1, w_dist=10.0, w_rep=1.0
        )
        assert val_total_terms(distill) == [
            "val_task_loss",
            "0.1 * val_kd_rank_loss",
            "10.0 * val_kd_dist_loss",
            "1.0 * val_kd_rep_loss",
        ]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_early_stopping.py -n0 -k kd_rank_rep -v`
Expected: FAIL with `KeyError: 'kd_rank_rep'`.

- [ ] **Step 3: Implement** — in `_VAL_TERMS` add after the `"kd_rank"` entry:

```python
    "kd_rank_rep": (
        ("w_rank", "val_kd_rank_loss"),
        ("w_dist", "val_kd_dist_loss"),
        ("w_rep", "val_kd_rep_loss"),
    ),
```

- [ ] **Step 4: Run the file**

Run: `.venv/bin/python -m pytest tests/eval/test_early_stopping.py -n0 -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src/eval/early_stopping.py tests/eval/test_early_stopping.py
git add src/eval/early_stopping.py tests/eval/test_early_stopping.py
git commit -m "feat(early-stopping): kd_rank_rep monitors rank, dist, and rep val terms"
```

---

### Task 3: Trainer gates — row bank rep term, context stream, main wiring

**Files:**
- Modify: `src/train_b0.py:114-115` (`_AUX_HEAD_ARMS` block), `:2851-2852` (`KDContextStream.__init__` guard), `:3296-3310` (`KDRowBank` staging), `:3412` (rep loss gate), `:3533` (rep telemetry gate), `:4907` (`main` context-stream construction)
- Test: `tests/test_train_b0_kd.py`

**Interfaces:**
- Consumes: Task 1's arm string.
- Produces: `_REP_COS_ARMS: frozenset[str]` module constant in `src/train_b0.py`; `KDRowBank.loss` returns `w_rep * kd_rep_loss(pair_repr, teacher_rep)` plus logit telemetry for a kd_rank_rep config; `KDContextStream(distill=...)` accepts any config with `w_rank > 0`.

- [ ] **Step 1: Write the failing tests** — append after `test_kd_rank_row_bank_is_telemetry_only` in `tests/test_train_b0_kd.py` (the file already imports `DistillConfig`, `KDRowBank`, `KDContextStream`, `load_kd_targets`, `np`, `torch`, `nn`, `pytest`, `Accelerator`, and defines `_write_targets`, `_ring_pairs`, `_context_fixture`, `_context_stream`):

```python
def test_kd_rank_rep_row_bank_emits_weighted_cosine_and_logit_telemetry(tmp_path: Path) -> None:
    node_ids = [f"n{i}" for i in range(4)]
    train_pairs = _ring_pairs(node_ids)
    train_labels = [1, 0, 1, 0]
    teacher_rep = np.eye(4, dtype=np.float32)
    _write_targets(
        tmp_path / "targets",
        node_ids=node_ids,
        train_pairs=train_pairs,
        train_labels=train_labels,
        teacher_logit=np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32),
        teacher_rep=teacher_rep,
    )
    model = nn.Module()
    model.d_model = 4  # type: ignore[assignment]
    bank = KDRowBank(
        DistillConfig(
            targets_path="t", context_targets_path="c", w_rank=0.1, w_dist=10.0, w_rep=2.0
        ),
        load_kd_targets(tmp_path / "targets"),
        train_pairs=train_pairs,
        train_labels=train_labels,
        val_pairs=train_pairs[:1],
        val_labels=train_labels[:1],
        model=model,
        device=torch.device("cpu"),
    )
    assert bank.arm == "kd_rank_rep"
    assert bank.train_rep is not None
    student_rep = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], requires_grad=True)
    output = {"logits": torch.tensor([0.3, -0.7]), "pair_repr": student_rep}
    loss, stats = bank.loss({"_row_id": torch.tensor([0, 1])}, output)
    expected = 2.0 * kd_rep_loss(student_rep, torch.tensor(teacher_rep[[0, 1]]))
    assert torch.allclose(loss, expected, atol=1e-6)
    assert loss.requires_grad
    assert stats["rows"] == 2.0
    assert "sum_rep_cos" in stats and "sum_st" in stats
    assert "sum_logit_bce" not in stats
    telemetry = bank.epoch_telemetry(Accelerator(cpu=True), stats)
    assert {"kd_rep_cos", "kd_rep_loss", "kd_logit_corr", "kd_prob_mae"} <= telemetry.keys()
    assert bank.global_relational is False
    assert bank.val_diagnostics().teacher_rep is not None


def test_context_stream_accepts_kd_rank_rep_and_rejects_arms_without_w_rank() -> None:
    targets, table = _context_fixture()

    def build(distill: DistillConfig) -> KDContextStream:
        return KDContextStream(
            distill,
            targets,
            table,
            allowed_nodes=frozenset(targets.node_ids),
            forbidden_internal_nodes=frozenset({"n0"}),
            epochs=2,
            rank=0,
            world_size=1,
            token_budget=1 << 20,
        )

    stream = build(
        DistillConfig(
            targets_path="rows", context_targets_path="contexts", w_rank=0.1, w_dist=10.0, w_rep=1.0
        )
    )
    assert stream is not None
    with pytest.raises(ValueError, match="w_rank"):
        build(DistillConfig(targets_path="rows", w_rep=1.0))
```

Add `kd_rep_loss` to the existing `from src.distill.losses import ...` line.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train_b0_kd.py -n0 -k "kd_rank_rep or rejects_arms_without_w_rank" -v`
Expected: first FAILs on `assert bank.train_rep is not None` (rep never staged); second FAILs with `ValueError: KDContextStream requires the active kd_rank arm`.

- [ ] **Step 3: Implement** — in `src/train_b0.py`:

After the `_AUX_HEAD_ARMS` definition (line 115) add:
```python
# Arms that align `pair_repr` to ``teacher_rep`` by per-row cosine (the kd_rep term).
_REP_COS_ARMS = frozenset({"kd_rep", "kd_rank_rep"})
```

`KDContextStream.__init__` guard (line 2851-2852) becomes:
```python
        if distill.w_rank <= 0.0:
            raise ValueError("KDContextStream requires an active w_rank term (kd_rank or kd_rank_rep)")
```

`KDRowBank.__init__` staging (lines 3301 and 3306):
```python
        self.train_rep: torch.Tensor | None = None
        if self.arm in {"kd_gram", "kd_gen"} or self.arm in _REP_COS_ARMS | _AUX_HEAD_ARMS:
            self.train_rep = torch.as_tensor(
                targets.teacher_rep, dtype=torch.float16, device=device
            )
        val_teacher_rep: torch.Tensor | None = None
        if self.arm == "kd_gram" or self.arm in _REP_COS_ARMS | _AUX_HEAD_ARMS:
            val_teacher_rep = torch.as_tensor(
                targets.val_teacher_rep, dtype=torch.float16, device=device
            )
```

`KDRowBank.loss` (line 3412): `if self.arm in _REP_COS_ARMS:` (body unchanged; the error string stays `"kd_rep requires the model forward to emit kd_rep or pair_repr"`).

`KDRowBank.epoch_telemetry` (line 3533): `if self.arm in _REP_COS_ARMS:`.

`main` (line 4907): `if cfg.distill.w_rank > 0.0:` replaces `if cfg.distill.arm == "kd_rank":`.

Docstrings: in `KDRowBank`'s class docstring change ``` ``kd_rank`` keeps this bank only for official-row ``` … to mention that `kd_rank_rep` adds the cosine term on the same rows; in `KDValDiagnostics.teacher_rep` change `populated for ``kd_rep``, ``kd_gram``, and ``kd_struct``` to `populated for the cosine arms, ``kd_gram``, and the aux-head arms`. Keep line counts.

- [ ] **Step 4: Run the KD trainer tests**

Run: `.venv/bin/python -m pytest tests/test_train_b0_kd.py -n0 -q`
Expected: all PASS (spawned DDP tests included; ~2–4 min).

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src/train_b0.py tests/test_train_b0_kd.py
git add src/train_b0.py tests/test_train_b0_kd.py
git commit -m "feat(train_b0): kd_rank_rep runs the kd_rep cosine term alongside the context stream"
```

---

### Task 4: Validation diagnostics report rep cosine and context terms in one outcome

**Files:**
- Modify: `src/train_b0.py:2323-2325` (`collect_diag`), `:2358` (rep branch), `:2484-2487` (metric keys)
- Test: `tests/test_train_b0_kd.py`

**Interfaces:**
- Consumes: `_REP_COS_ARMS` (Task 3), `KDValDiagnostics(arm="kd_rank_rep", teacher_rep=..., context_stream=...)`.
- Produces: `_evaluate_distributed(...).kd` containing `val_kd_rep_cos`, `val_kd_rep_loss`, `val_kd_logit_corr`, `val_kd_logit_loss`, `val_kd_prob_mae`, `val_kd_rank_loss`, `val_kd_dist_loss` for a kd_rank_rep run — the keys Task 2's monitor needs.

- [ ] **Step 1: Write the failing test** — after `test_evaluate_distributed_kd_rep_diagnostics`:

```python
class _RankRepDiagModel(nn.Module):
    """Answers cls-row batches like `_RepDiagModel` and context batches like `_ContextToy`."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "emb_a" in batch:
            scaled_a = (self.scale * batch["emb_a"]).sum(dim=(1, 2))
            scaled_b = (self.scale * batch["emb_b"]).sum(dim=(1, 2))
            return {"logits": scaled_a - scaled_b}
        return {"logits": self.scale * batch["logit_seed"], "pair_repr": self.scale * batch["rep_seed"]}


def test_evaluate_distributed_kd_rank_rep_reports_rep_and_context_diagnostics() -> None:
    targets, table = _context_fixture()
    stream = _context_stream(targets, table)
    n, rep_dim = 3, 4
    rep_seed = torch.randn(n, rep_dim)
    batch = {
        "label": torch.tensor([1.0, 0.0, 1.0]),
        "_row_id": torch.tensor([0, 1, 2]),
        "logit_seed": torch.tensor([0.5, -0.5, 1.0]),
        "rep_seed": rep_seed,
    }
    teacher_logit = torch.tensor([0.4, -0.6, 0.9])
    kd_val = KDValDiagnostics(
        arm="kd_rank_rep",
        teacher_logit=teacher_logit,
        teacher_logit_np=teacher_logit.double().numpy(),
        teacher_rep=rep_seed.clone().to(torch.float16),
        teacher_latent=None,
        context_stream=stream,
    )
    outcome = _evaluate_distributed(
        _RankRepDiagModel(), [batch], Accelerator(cpu=True), expected_row_ids=np.arange(n), kd_val=kd_val
    )
    assert outcome.kd is not None
    assert {
        "val_kd_rep_cos",
        "val_kd_rep_loss",
        "val_kd_logit_corr",
        "val_kd_logit_loss",
        "val_kd_prob_mae",
        "val_kd_rank_loss",
        "val_kd_dist_loss",
    } <= outcome.kd.keys()
    assert outcome.kd["val_kd_rep_cos"] == pytest.approx(1.0, abs=1e-3)
    distill = DistillConfig(
        targets_path="t", context_targets_path="c", w_rank=0.1, w_dist=10.0, w_rep=1.0
    )
    assert compose_val_total(outcome.task_loss, outcome.kd, distill) >= outcome.task_loss
```

(`compose_val_total`, `KDValDiagnostics`, `_evaluate_distributed` are already imported in this test file.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_b0_kd.py -n0 -k kd_rank_rep_reports -v`
Expected: FAIL — `val_kd_rep_cos` missing from `outcome.kd`.

- [ ] **Step 3: Implement** — in `_evaluate_distributed`:

```python
    collect_diag = kd_val is not None and (
        kd_val.arm in _REP_COS_ARMS or kd_val.arm in _AUX_HEAD_ARMS or inject_latent
    )
```
Line 2358: `if kd_val.arm in _REP_COS_ARMS:` (body unchanged).
Lines 2484-2486:
```python
                key = "val_kd_rep_cos" if kd_val.arm in _REP_COS_ARMS else "val_kd_latent_cos"
                kd_metrics[key] = float(diag_np.mean())
                if kd_val.arm in _REP_COS_ARMS:
```

- [ ] **Step 4: Run the file**

Run: `.venv/bin/python -m pytest tests/test_train_b0_kd.py -n0 -q`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src/train_b0.py tests/test_train_b0_kd.py
git add src/train_b0.py tests/test_train_b0_kd.py
git commit -m "feat(train_b0): kd_rank_rep validation diagnostics cover rep cosine and context terms"
```

---

### Task 5: Parametrize the kd_rank sweep driver with a `SweepSpec`

**Files:**
- Modify: `src/experiments/kd_rank_strict_hpo.py` (`materialize_trial_config`, `build_study`, `enqueue_priors`, `run_sweep`, `print_report`, new `SweepSpec` + `_write_trial_config` + `KD_RANK_SPEC`)
- Test: `tests/experiments/test_kd_rank_strict_hpo.py` (must pass unchanged)

**Interfaces:**
- Produces, in `src/experiments/kd_rank_strict_hpo.py`:
  ```python
  @dataclass(frozen=True)
  class SweepSpec:
      study_name: str
      n_startup_trials: int
      priors: tuple[dict[str, object], ...]
      param_names: tuple[str, ...]
      suggest: Callable[[optuna.Trial], dict[str, object]]
      materialize: Callable[[Path, Mapping[str, object], int, Path], Path]
      prepare: Callable[[argparse.Namespace], None]

  def _write_trial_config(base_config: Path, distill_overrides: Mapping[str, object], trial_number: int, sweep_dir: Path) -> Path
  def build_study(db_path: Path, *, study_name: str = STUDY_NAME, n_startup_trials: int = N_STARTUP_TRIALS) -> optuna.Study
  def enqueue_priors(study: optuna.Study, priors: Sequence[Mapping[str, object]] = ENQUEUED_PRIORS) -> None
  def run_sweep(args: argparse.Namespace, spec: SweepSpec = KD_RANK_SPEC) -> None
  def print_report(study: optuna.Study, param_names: Sequence[str] = ("w_rank", "w_dist", "bank", "margin")) -> None
  KD_RANK_SPEC: SweepSpec
  ```
  `run_sweep` reads `args.base_config`, `args.sweep_dir`, `args.n_trials`, `args.rd_band` and calls `spec.prepare(args)` where it used to call `dump_missing_banks(args)`.

- [ ] **Step 1: Run the existing driver tests as the regression baseline**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_strict_hpo.py -n0 -q`
Expected: all PASS before the change.

- [ ] **Step 2: Implement** — in `src/experiments/kd_rank_strict_hpo.py`:

Add `from collections.abc import Callable, Mapping, Sequence` (extend the existing import). Split `materialize_trial_config`:

```python
def _write_trial_config(
    base_config: Path, distill_overrides: Mapping[str, object], trial_number: int, sweep_dir: Path
) -> Path:
    """Write trial ``trial_number``'s config: base + ``distill_overrides`` + ``output_dir``.

    Raises:
        ValueError: If the resulting ``distill`` section is illegal.
    """
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["output_dir"] = str(sweep_dir / f"trial_{trial_number:03d}")
    distill = {**cfg["distill"], **distill_overrides}
    cfg["distill"] = distill
    DistillConfig.from_mapping(distill)
    config_path = sweep_dir / "configs" / f"trial_{trial_number:03d}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def materialize_trial_config(
    base_config: Path, params: Mapping[str, object], trial_number: int, sweep_dir: Path
) -> Path:
    """Write trial ``trial_number``'s kd_rank config; only the five whitelisted keys differ.

    Raises:
        KeyError: On an unknown bank name.
        ValueError: If the resulting ``distill`` section is illegal.
    """
    overrides = {
        "w_rank": float(params["w_rank"]),  # type: ignore[arg-type]
        "w_dist": float(params["w_dist"]),  # type: ignore[arg-type]
        "margin": float(params["margin"]),  # type: ignore[arg-type]
        "context_targets_path": BANKS[str(params["bank"])].path,
    }
    return _write_trial_config(base_config, overrides, trial_number, sweep_dir)
```

Rewrite the three parametrized helpers:

```python
def build_study(
    db_path: Path, *, study_name: str = STUDY_NAME, n_startup_trials: int = N_STARTUP_TRIALS
) -> optuna.Study:
    """Create-or-load a sweep study (kd_rank's by default)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        seed=0,
        multivariate=True,
        n_startup_trials=n_startup_trials,
        constraints_func=_constraints,
    )
    return optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        directions=["maximize", "minimize"],
        sampler=sampler,
        load_if_exists=True,
    )


def enqueue_priors(
    study: optuna.Study, priors: Sequence[Mapping[str, object]] = ENQUEUED_PRIORS
) -> None:
    """Enqueue every prior that has no waiting, running, or completed trial.

    A prior whose trial failed is re-enqueued, so a fixed environment gets
    the full prior set on restart (``skip_if_exists`` would treat the failed
    twin as done).
    """
    live = (TrialState.WAITING, TrialState.RUNNING, TrialState.COMPLETE)
    seen = [
        t.system_attrs.get("fixed_params", t.params)
        for t in study.get_trials(deepcopy=False, states=live)
    ]
    for params in priors:
        if dict(params) not in seen:
            study.enqueue_trial(dict(params))


def print_report(
    study: optuna.Study, param_names: Sequence[str] = ("w_rank", "w_dist", "bank", "margin")
) -> None:
    """Print the full trial table, then the feasible Pareto front."""
    columns = [
        "auprc",
        "gs",
        "rd",
        "degree_mmd",
        "clustering_mmd",
        "spectral_mmd",
        "selected_epoch",
    ]
    print("number state " + " ".join(param_names) + " " + " ".join(columns))  # noqa: T201 -- CLI report goes to stdout
    for t in study.get_trials(deepcopy=False):
        surface = t.user_attrs.get("surface", {})
        params = " ".join(str(t.params.get(name, "-")) for name in param_names)
        values = " ".join(f"{surface[c]:.4f}" if c in surface else "-" for c in columns)
        print(f"{t.number} {t.state.name} {params} {values}")  # noqa: T201 -- CLI report goes to stdout
    front = ", ".join(str(t.number) for t in study.best_trials)
    print(f"feasible Pareto front (advisory): trials [{front}]")  # noqa: T201 -- CLI report goes to stdout
```

Add the `SweepSpec` dataclass (after `TrialOutcome`):

```python
@dataclass(frozen=True)
class SweepSpec:
    """One arm's sweep: study identity, priors, search space, config writer, and prep step."""

    study_name: str
    n_startup_trials: int
    priors: tuple[dict[str, object], ...]
    param_names: tuple[str, ...]
    suggest: Callable[[optuna.Trial], dict[str, object]]
    materialize: Callable[[Path, Mapping[str, object], int, Path], Path]
    prepare: Callable[[argparse.Namespace], None]
```

Rewrite `run_sweep` (the `KD_RANK_SPEC` constant sits directly above it):

```python
def run_sweep(args: argparse.Namespace, spec: SweepSpec = KD_RANK_SPEC) -> None:
    """Drive the whole sweep: reconcile, prepare banks, ask/tell until budget."""
    study = build_study(
        args.sweep_dir / "optuna.db",
        study_name=spec.study_name,
        n_startup_trials=spec.n_startup_trials,
    )
    reconcile_running(study, args.sweep_dir, args.rd_band)
    enqueue_priors(study, spec.priors)
    spec.prepare(args)
    failures = 0
    while _n_complete(study) < args.n_trials:
        trial = study.ask()
        params = spec.suggest(trial)
        config_path = spec.materialize(args.base_config, params, trial.number, args.sweep_dir)
        run_command(["bash", "hpc/run.sh", "train", str(config_path), "--skip-test"])
        run_dir = args.sweep_dir / f"trial_{trial.number:03d}"
        try:
            outcome = trial_outcome(run_dir, args.rd_band)
        except RunFailure:
            study.tell(trial, state=TrialState.FAIL)
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{failures} consecutive failed trials, last {run_dir}: fix the "
                    "environment and restart the sweep"
                ) from None
            continue
        failures = 0
        trial.set_user_attr("constraint", [outcome.constraint])
        trial.set_user_attr("surface", outcome.surface)
        study.tell(trial, values=[outcome.gs, outcome.geo_mmd])
    print_report(study, spec.param_names)
```

Define, after `dump_missing_banks` and before `run_sweep`:
```python
KD_RANK_SPEC = SweepSpec(
    study_name=STUDY_NAME,
    n_startup_trials=N_STARTUP_TRIALS,
    priors=ENQUEUED_PRIORS,
    param_names=("w_rank", "w_dist", "bank", "margin"),
    suggest=suggest_params,
    materialize=materialize_trial_config,
    prepare=dump_missing_banks,
)
```
(`KD_RANK_SPEC` must be defined before `run_sweep`'s default argument is evaluated, so place it above `run_sweep`.)

- [ ] **Step 3: Run the existing driver tests unchanged**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_strict_hpo.py -n0 -q`
Expected: all PASS with zero test edits (`hpo.run_sweep(args)`, `hpo.build_study(db)`, `hpo.enqueue_priors(study)` still work through the defaults; monkeypatched `hpo.run_command`/`hpo.BANKS` still take effect because bodies read module globals at call time).

- [ ] **Step 4: Lint and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src/experiments/kd_rank_strict_hpo.py tests/experiments/test_kd_rank_strict_hpo.py
git add src/experiments/kd_rank_strict_hpo.py
git commit -m "refactor(hpo): parametrize the strict sweep driver with a SweepSpec"
```

---

### Task 6: `kd_rank_rep` driver, base config, tests

**Files:**
- Create: `src/experiments/kd_rank_rep_hpo.py`, `configs/autoresearch/kd_rank_rep.yaml`, `tests/experiments/test_kd_rank_rep_hpo.py`

**Interfaces:**
- Consumes: `SweepSpec`, `_write_trial_config`, `BANKS`, `build_study`, `run_sweep`, `run_command` from Task 5.
- Produces:
  ```python
  STUDY_NAME = "kd_rank_rep_strict"; N_STARTUP_TRIALS = 4
  ENQUEUED_PRIORS: tuple[dict[str, object], ...]   # 4 rows from the spec
  def suggest_params(trial: optuna.Trial) -> dict[str, object]           # w_rank, w_dist, w_rep
  def materialize_trial_config(base_config, params, trial_number, sweep_dir, *, bank: str, margin: float) -> Path
  def require_bank(args: argparse.Namespace) -> None                     # fails closed on a missing manifest
  def build_spec(args: argparse.Namespace) -> SweepSpec
  def build_parser() -> argparse.ArgumentParser
  def main(argv: Sequence[str] | None = None) -> None
  ```

- [ ] **Step 1: Write the base config** `configs/autoresearch/kd_rank_rep.yaml` — copy `configs/autoresearch/kd_rank.yaml` verbatim, then replace the three header comment lines with:

```yaml
# Autoresearch sweep surface for kd_rank_rep (strict-LLP rank/dist + kd_rep cosine):
# cadence-2 validation, --skip-test per trial. The driver rewrites only
# distill.w_rank/w_dist/w_rep/margin/context_targets_path and output_dir.
```
and the tail with:
```yaml
output_dir: outputs/b1_kd_rank_rep_hpo/trial_000
mixed_precision: "bf16"
distill:
  targets_path: outputs/distill/kd_row_targets_breadth_first
  context_targets_path: outputs/distill/kd_ctx_targets_breadth_first_h2ns3
  w_rank: 0.1
  w_dist: 10.0
  w_rep: 0.1
  margin: 0.1
```
Everything between (model, data, optim, `eval.topology_every: 2`, runtime, seed) stays identical to `kd_rank.yaml`.

- [ ] **Step 2: Write the failing tests** — `tests/experiments/test_kd_rank_rep_hpo.py`:

```python
"""CPU-only tests for the kd_rank_rep strict Optuna sweep driver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml
from optuna.trial import TrialState
from src.distill.config import DistillConfig
from src.experiments import kd_rank_rep_hpo as hpo
from src.experiments import kd_rank_strict_hpo as shared

from tests.autoresearch.conftest import make_cadence_rows

pytestmark = pytest.mark.unit

BASE_CONFIG = Path("configs/autoresearch/kd_rank_rep.yaml")


def test_base_config_is_a_legal_kd_rank_rep_arm() -> None:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    assert DistillConfig.from_mapping(cfg["distill"]).arm == "kd_rank_rep"
    assert cfg["eval"]["topology_every"] == 2


def test_enqueued_priors_match_spec() -> None:
    assert hpo.ENQUEUED_PRIORS == (
        {"w_rank": 0.1, "w_dist": 10.0, "w_rep": 0.1},
        {"w_rank": 0.1, "w_dist": 10.0, "w_rep": 1.0},
        {"w_rank": 0.1, "w_dist": 10.0, "w_rep": 10.0},
        {"w_rank": 1.0, "w_dist": 1.0, "w_rep": 1.0},
    )
    assert hpo.N_STARTUP_TRIALS == len(hpo.ENQUEUED_PRIORS)


def test_suggest_params_covers_the_three_log_boxes(tmp_path: Path) -> None:
    study = shared.build_study(tmp_path / "optuna.db", study_name=hpo.STUDY_NAME, n_startup_trials=1)
    params = hpo.suggest_params(study.ask())
    assert set(params) == {"w_rank", "w_dist", "w_rep"}
    assert 0.01 <= params["w_rank"] <= 1.0
    assert 0.1 <= params["w_dist"] <= 100.0
    assert 0.01 <= params["w_rep"] <= 100.0


def test_materialize_changes_only_whitelisted_keys(tmp_path: Path) -> None:
    params = {"w_rank": 0.3, "w_dist": 5.0, "w_rep": 2.0}
    config_path = hpo.materialize_trial_config(BASE_CONFIG, params, 3, tmp_path, bank="h2ns5", margin=0.2)
    assert config_path == tmp_path / "configs" / "trial_003.yaml"
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    trial = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert trial["output_dir"] == str(tmp_path / "trial_003")
    assert trial["distill"]["w_rank"] == 0.3
    assert trial["distill"]["w_dist"] == 5.0
    assert trial["distill"]["w_rep"] == 2.0
    assert trial["distill"]["margin"] == 0.2
    assert trial["distill"]["context_targets_path"] == shared.BANKS["h2ns5"].path
    trial["output_dir"] = base["output_dir"]
    trial["distill"] = base["distill"]
    assert trial == base


def test_materialize_rejects_unknown_bank_and_zero_weight(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        hpo.materialize_trial_config(
            BASE_CONFIG, {"w_rank": 0.1, "w_dist": 1.0, "w_rep": 1.0}, 1, tmp_path, bank="h9", margin=0.1
        )
    with pytest.raises(ValueError):
        hpo.materialize_trial_config(
            BASE_CONFIG, {"w_rank": 0.1, "w_dist": 1.0, "w_rep": 0.0}, 1, tmp_path, bank="h2ns3", margin=0.1
        )


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    argv = ["--sweep-dir", str(tmp_path), "--n-trials", "2"]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return hpo.build_parser().parse_args(argv)


def test_parser_defaults_match_spec() -> None:
    args = hpo.build_parser().parse_args([])
    assert args.base_config == BASE_CONFIG
    assert args.sweep_dir == Path("outputs/b1_kd_rank_rep_hpo")
    assert (args.n_trials, args.rd_band, args.bank, args.margin) == (12, 0.05, "h2ns3", 0.1)


def test_require_bank_fails_closed_without_a_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = shared.BANKS["h2ns3"]
    monkeypatch.setitem(
        shared.BANKS, "h2ns3", shared.BankSpec(spec.rw_step, spec.hops, spec.ns_rate, str(tmp_path / "bank"))
    )
    with pytest.raises(RuntimeError, match="h2ns3"):
        hpo.require_bank(_args(tmp_path))
    (tmp_path / "bank").mkdir()
    (tmp_path / "bank" / "manifest.json").write_text("{}", encoding="utf-8")
    hpo.require_bank(_args(tmp_path))


def _publish_run(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(cfg["output_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in make_cadence_rows()), encoding="utf-8"
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"selected_epoch": 2, "arm": "kd_rank_rep", "config_hash": "d", "checkpoint_id": "c"}),
        encoding="utf-8",
    )
    (run_dir / "complete.json").write_text(
        json.dumps({"status": "complete", "attempt_id": "fixture", "total_seconds": 60.0}),
        encoding="utf-8",
    )


def test_main_runs_priors_first_with_the_frozen_bank_and_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = shared.BANKS["h2ns3"]
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    (bank_dir / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(
        shared.BANKS, "h2ns3", shared.BankSpec(spec.rw_step, spec.hops, spec.ns_rate, str(bank_dir))
    )
    launched: list[Path] = []

    def fake_run(cmd: list[str]) -> int:
        assert cmd[:3] == ["bash", "hpc/run.sh", "train"] and cmd[-1] == "--skip-test"
        launched.append(Path(cmd[3]))
        _publish_run(Path(cmd[3]))
        return 0

    monkeypatch.setattr(shared, "run_command", fake_run)
    hpo.main(["--sweep-dir", str(tmp_path), "--n-trials", "2"])
    study = shared.build_study(tmp_path / "optuna.db", study_name=hpo.STUDY_NAME)
    complete = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    assert [t.params for t in complete] == [dict(p) for p in hpo.ENQUEUED_PRIORS[:2]]
    for config_path in launched:
        distill = yaml.safe_load(config_path.read_text(encoding="utf-8"))["distill"]
        assert distill["context_targets_path"] == str(bank_dir)
        assert distill["margin"] == 0.1
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_rep_hpo.py -n0 -v`
Expected: collection error `ModuleNotFoundError: No module named 'src.experiments.kd_rank_rep_hpo'`.

- [ ] **Step 4: Implement** `src/experiments/kd_rank_rep_hpo.py`:

```python
"""Unattended Optuna sweep for the ``kd_rank_rep`` double-KD arm.

Same ask-and-tell constrained MO-TPE loop, objectives (GS max, geometric-mean
MMD ratio min), and ``|log RD|`` soft constraint as the strict-LLP kd_rank
sweep (`src.experiments.kd_rank_strict_hpo`); this study searches only the
three loss weights and inherits the kd_rank winner's context bank and margin
through ``--bank``/``--margin``. Winner selection stays the frozen
five-metric undominated verdict plus the human pick.
Spec: ``docs/superpowers/specs/2026-09-04-kd-rank-rep-double-kd-design.md``.
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Mapping, Sequence
from pathlib import Path

import optuna

from src.experiments.kd_rank_strict_hpo import BANKS, SweepSpec, _write_trial_config, run_sweep

STUDY_NAME = "kd_rank_rep_strict"
N_STARTUP_TRIALS = 4

ENQUEUED_PRIORS: tuple[dict[str, object], ...] = (
    {"w_rank": 0.1, "w_dist": 10.0, "w_rep": 0.1},
    {"w_rank": 0.1, "w_dist": 10.0, "w_rep": 1.0},
    {"w_rank": 0.1, "w_dist": 10.0, "w_rep": 10.0},
    {"w_rank": 1.0, "w_dist": 1.0, "w_rep": 1.0},
)


def suggest_params(trial: optuna.Trial) -> dict[str, object]:
    """Draw one point of the three-weight log box (enqueued values pass through)."""
    return {
        "w_rank": float(trial.suggest_float("w_rank", 0.01, 1.0, log=True)),
        "w_dist": float(trial.suggest_float("w_dist", 0.1, 100.0, log=True)),
        "w_rep": float(trial.suggest_float("w_rep", 0.01, 100.0, log=True)),
    }


def materialize_trial_config(
    base_config: Path,
    params: Mapping[str, object],
    trial_number: int,
    sweep_dir: Path,
    *,
    bank: str,
    margin: float,
) -> Path:
    """Write trial ``trial_number``'s config; only the whitelisted distill keys differ.

    Raises:
        KeyError: On an unknown bank name.
        ValueError: If the resulting ``distill`` section is illegal.
    """
    overrides = {
        "w_rank": float(params["w_rank"]),  # type: ignore[arg-type]
        "w_dist": float(params["w_dist"]),  # type: ignore[arg-type]
        "w_rep": float(params["w_rep"]),  # type: ignore[arg-type]
        "margin": float(margin),
        "context_targets_path": BANKS[bank].path,
    }
    return _write_trial_config(base_config, overrides, trial_number, sweep_dir)


def require_bank(args: argparse.Namespace) -> None:
    """Fail closed before any training budget if the frozen bank is not on disk.

    Raises:
        RuntimeError: If the bank's manifest is missing (dump it with the
            kd_rank sweep driver first).
    """
    path = Path(BANKS[args.bank].path)
    if not (path / "manifest.json").exists():
        raise RuntimeError(f"context bank {args.bank} has no manifest at {path}")


def build_spec(args: argparse.Namespace) -> SweepSpec:
    """Bind the frozen bank and margin into the sweep specification."""
    return SweepSpec(
        study_name=STUDY_NAME,
        n_startup_trials=N_STARTUP_TRIALS,
        priors=ENQUEUED_PRIORS,
        param_names=("w_rank", "w_dist", "w_rep"),
        suggest=suggest_params,
        materialize=functools.partial(materialize_trial_config, bank=args.bank, margin=args.margin),
        prepare=require_bank,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the `python -m src.experiments.kd_rank_rep_hpo` parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config", type=Path, default=Path("configs/autoresearch/kd_rank_rep.yaml")
    )
    parser.add_argument("--sweep-dir", type=Path, default=Path("outputs/b1_kd_rank_rep_hpo"))
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--rd-band", type=float, default=0.05)
    parser.add_argument("--bank", choices=sorted(BANKS), default="h2ns3")
    parser.add_argument("--margin", type=float, default=0.1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the unattended container sweep."""
    args = build_parser().parse_args(argv)
    run_sweep(args, build_spec(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run both driver test files**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_rep_hpo.py tests/experiments/test_kd_rank_strict_hpo.py -n0 -q`
Expected: all PASS.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src/experiments tests/experiments
git add src/experiments/kd_rank_rep_hpo.py configs/autoresearch/kd_rank_rep.yaml tests/experiments/test_kd_rank_rep_hpo.py
git commit -m "feat(hpo): kd_rank_rep strict Optuna sweep driver and base config"
```

---

### Task 7: Documentation and full verification

**Files:**
- Modify: `docs/03-experiments.md:69-71` (§1.4 table), `:83-85` (§1.5 HPO paragraph); `hpc/README.md:244-245`

- [ ] **Step 1: §1.4 row** — insert after the `kd_representation` row (line 70):

```
| kd_rank_rep | strict-LLP rank + distribution KD plus per-row pair-representation cosine | PMA(4) Full-Ego Oracle context bank + row bank | `w_rank` log-uniform [0.01, 1]; `w_dist` log-uniform [0.1, 100]; `w_rep` log-uniform [0.01, 100]; bank and margin inherited from kd_ranking | joint logit + representation transfer |
```

- [ ] **Step 2: §1.5 sentence** — replace the three-line HPO paragraph (starts `HPO: a Phase-0 grid`) with these three lines:

```
HPO: a Phase-0 grid (24 runs) fixed per-arm incumbents (kd_logit_w100, kd_rank_wr0p1_wd1, kd_gram_w1, kd_rep_w0p1); kd_rank
continues with a 16-trial constrained MO-TPE study (GS ↑, geometric-mean MMD ↓, soft constraint `|log RD| <= 0.05`) over w_rank ×
w_dist × context bank × margin, and kd_rank_rep with a 12-trial study of the same form over w_rank × w_dist × w_rep at the inherited bank and margin. The best configuration per arm runs the held-out test protocol exactly once; provenance and HPC completion are rules 5--6 (§5).
```

- [ ] **Step 3: Runbook line** — replace `hpc/README.md` lines 244-245 with two lines:

```
# unattended strict sweeps: kd_rank dumps missing context banks then runs 16 trials; kd_rank_rep runs 12 at the frozen bank/margin
.venv/bin/python -m src.experiments.kd_rank_strict_hpo --teacher-checkpoint <full_ego_oracle best.pt>   # or: -m src.experiments.kd_rank_rep_hpo --bank h2ns3 --margin 0.1
```

- [ ] **Step 4: Full fast test suite, lint, and mypy**

```bash
.venv/bin/python -m pytest -m "not slow and not integration" -q
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format --check src tests
.venv/bin/python -m mypy src tests 2>&1 | tail -1     # must still report exactly 98 errors in 14 files
```
Expected: pytest all PASS; ruff clean; mypy count unchanged.

- [ ] **Step 5: Commit**

```bash
git add docs/03-experiments.md hpc/README.md
git commit -m "docs: record the kd_rank_rep arm and its 12-trial strict sweep"
```

---

## Operator steps after merge (H20, not part of this plan's code)

1. `git push`, then `git pull` on the 30838 checkout (shared with 30030).
2. Confirm `outputs/distill/kd_ctx_targets_breadth_first_h2ns3/manifest.json` exists (dumped by the kd_rank sweep).
3. If the kd_rank sweep has finished with a different bank/margin winner, pass them as `--bank`/`--margin`.
4. Launch on a free container:
   ```bash
   mkdir -p outputs/logs && OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 nohup .venv/bin/python -m src.experiments.kd_rank_rep_hpo > outputs/logs/kd_rank_rep_hpo.log 2>&1 &
   ```
5. After 12 completed trials: five-metric undominated verdict + human pick; `hpc/run.sh test` on the winner's `outputs/b1_kd_rank_rep_hpo/trial_NNN`; write `docs/results/kd_rank_rep_hpo/README.md` in the kd_rank sweep format; compare against `kd_control` and the matched-epoch controls.
