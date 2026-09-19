"""Durable, one-shot queue for the motif-prompt v17 campaign (spec section 7.6).

The queue is deliberately serial.  It can be launched behind another queue or
training process, waits for that exact Linux process and for this container's
compute GPUs to become idle, and then owns the campaign through the first
terminal scientific decision.  Held-out test is opened only after a lane clears
the preregistered V_val adoption rule and its seed-1/2 replications publish.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from src.autoresearch.metrics_io import read_run
from src.eval.checkpoint_selection import CheckpointCandidate, select_checkpoint
from src.eval.fixed_threshold import evaluate_fixed_threshold, select_fixed_threshold
from src.eval.graph_metrics import MMDConfig
from src.eval.report_edge_metrics import report_edge_metrics
from src.experiments.g1_hardened_e2 import (
    _BENCHMARK_SUBDIR,
    load_test_graph,
    load_test_node_buckets,
)
from src.experiments.motif_crossfit import NoPassingPilotError, select_pilot_checkpoint
from src.experiments.motif_density_control import density_matched_report
from src.score_universe import _load_val_region_split, load_scores, validate_artifact_precision

CONFIG_ROOT = Path("configs/split_seed42")
OUTPUT_ROOT = Path("outputs/split_seed42")
CAMPAIGN_ROOT = OUTPUT_ROOT / "motif_v17"
WAVE3_CHECKPOINT = OUTPUT_ROOT / "motif_prompt_stage2_v3_prefix/best.pt"
PREFIX_BASE_RUN = OUTPUT_ROOT / "prefix_base"
POLL_SECONDS = 30.0
INTERVENTIONS = ("gates_off", "mean", "shuffle_graph")
FINAL_INTERVENTIONS = ("gates_off", "mean", "shuffle_graph", "permute_closure", "rewire_bridge")

FLAT_FOLDS = ("motif_crossfit_flat_fold0", "motif_crossfit_flat_fold1")
PRESENCE_GENERATORS = (
    "motif_prompt_stage2_v4_prefix",
    "motif_crossfit_presence_fold0",
    "motif_crossfit_presence_fold1",
)
LANES = {
    "shift": "motif_prompt_stage1_shift",
    "confidence": "motif_prompt_stage1_shift_conf",
    "control": "motif_prompt_stage1_shift_control",
}


def process_identity(pid: int) -> dict[str, object] | None:
    """Return stable identity for a live non-zombie Linux process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        if stat[0] == "Z":
            return None
        command = (
            Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        )
        return {"pid": pid, "start_ticks": stat[19], "command": command}
    except FileNotFoundError:
        return None


def same_process(identity: dict[str, object]) -> bool:
    """Reject PID reuse by comparing the kernel start time as well as PID."""
    current = process_identity(cast(int, identity["pid"]))
    return current is not None and current["start_ticks"] == identity["start_ticks"]


def gpu_compute_pids() -> list[int]:
    """Read compute PIDs from this container only."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    )
    return [int(value.strip()) for value in output.splitlines() if value.strip()]


def visible_gpu_count() -> int:
    """Count GPUs visible in this container without assuming a particular port."""
    output = subprocess.check_output(["nvidia-smi", "--list-gpus"], text=True)
    count = sum(bool(line.strip()) for line in output.splitlines())
    if count < 1:
        raise RuntimeError("no visible GPU")
    return count


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Replace a JSON status atomically so observers never see a partial write."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def config_output(config: Path) -> Path:
    """Read and validate the publication directory declared by a config."""
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("output_dir"), str):
        raise ValueError(f"{config}: missing string output_dir")
    return Path(payload["output_dir"])


def require_fresh(path: Path) -> None:
    """Never resume into or overwrite an existing publication/analysis path."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


class Queue:
    """Stateful v17 campaign driver with injectable execution for tests."""

    def __init__(
        self,
        *,
        root: Path,
        predecessor: dict[str, object],
        data_root: Path,
        pack_dir: Path,
        run_command: Callable[[list[str], Path], None] | None = None,
    ) -> None:
        self.root = root
        self.predecessor = predecessor
        self.data_root = data_root
        self.pack_dir = pack_dir
        self.status_path = root / "status.json"
        self.generator_runs: dict[str, tuple[Path, Path]] = {}
        self.deployed_generators: dict[str, Path] = {"flat": WAVE3_CHECKPOINT}
        self.caches: dict[str, Path] = {}
        self.deployed_presence_checkpoint: Path | None = None
        self.lane_configs: dict[str, Path] = {}
        self.state: dict[str, object] = {
            "schema": "motif_v17_queue_v1",
            "pid": os.getpid(),
            "predecessor": predecessor,
            "started_utc": datetime.now(UTC).isoformat(),
            "completed_stages": [],
        }
        self._executor = run_command or self._subprocess

    def status(self, value: str, *, stage: str, **extra: object) -> None:
        """Persist one observable queue transition."""
        self.state.update(
            status=value, stage=stage, updated_utc=datetime.now(UTC).isoformat(), **extra
        )
        atomic_json(self.status_path, self.state)

    @staticmethod
    def _subprocess(command: list[str], log: Path) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x", encoding="utf-8") as handle:
            subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
                env={**os.environ, "OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16"},
            )

    def command(self, stage: str, argv: list[str]) -> None:
        """Run one logged foreground command and record its completion."""
        self.status("running", stage=stage, command=argv)
        self._executor(argv, self.root / "logs" / f"{stage}.log")
        cast(list[str], self.state["completed_stages"]).append(stage)
        self.status("running", stage=stage)

    def wait(self) -> None:
        """Wait for the exact predecessor and all local compute jobs."""
        while True:
            predecessor_alive = same_process(self.predecessor)
            gpu_pids = gpu_compute_pids()
            self.status(
                "waiting",
                stage="predecessor_and_gpus",
                predecessor_alive=predecessor_alive,
                gpu_compute_pids=gpu_pids,
            )
            if not predecessor_alive and not gpu_pids:
                return
            time.sleep(POLL_SECONDS)

    def train(self, name: str) -> Path:
        """Publish one fresh training config, failing on collisions."""
        config = CONFIG_ROOT / f"{name}.yaml"
        return self.train_config(name, config)

    def train_config(self, name: str, config: Path) -> Path:
        """Publish one explicit config path, failing on output collisions."""
        output = config_output(config)
        require_fresh(output)
        self.command(f"train_{name}", ["bash", "hpc/run.sh", "train", str(config), "--skip-test"])
        if not (output / "complete.json").exists() or not (output / "best.pt").exists():
            raise RuntimeError(f"training exited without publication artifacts: {output}")
        return output

    def replicate(self, name: str, seed: int, cache: Path | None) -> Path:
        """Materialize and train a seed-only copy of the adopted lane config."""
        source = self.lane_configs.get(name, CONFIG_ROOT / f"{name}.yaml")
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: expected a YAML mapping")
        replica_name = f"{name}_seed{seed}"
        payload["seed"] = seed
        payload["output_dir"] = str(OUTPUT_ROOT / replica_name)
        if cache is not None:
            model = cast(dict[str, Any], payload["model"])
            model_config = cast(dict[str, Any], model["config"])
            motif = cast(dict[str, Any], model_config["motif_prompt"])
            corruption = cast(dict[str, Any], motif["corruption"])
            corruption["predicted_cache_path"] = str(cache)
        generated = self.root / "configs" / f"{replica_name}.yaml"
        generated.parent.mkdir(parents=True, exist_ok=True)
        require_fresh(generated)
        generated.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return self.train_config(replica_name, generated)

    def select_generator(self, name: str, run: Path) -> Path:
        """Run pilot B on every epoch and select its documented graph-only checkpoint."""
        config = CONFIG_ROOT / f"{name}.yaml"
        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        stage1 = Path(payload["model"]["config"]["motif_prompt"]["bundle_checkpoint"])
        checkpoints = sorted(run.glob("attempts/*/checkpoints/epoch-*.pt"))
        if len(checkpoints) != 8:
            raise RuntimeError(
                f"{run}: expected eight graph-only epoch checkpoints, got {len(checkpoints)}"
            )
        reports: list[Path] = []
        for checkpoint in checkpoints:
            output = self.root / "pilot_b" / name / checkpoint.stem
            require_fresh(output)
            self.command(
                f"pilot_{name}_{checkpoint.stem}",
                [
                    sys.executable,
                    "-m",
                    "src.experiments.motif_pilot_b",
                    "--checkpoint",
                    str(checkpoint),
                    "--config",
                    str(config),
                    "--stage1-checkpoint",
                    str(stage1),
                    "--pack-dir",
                    str(self.pack_dir),
                    "--data-root",
                    str(self.data_root),
                    "--strategy",
                    "breadth_first",
                    "--output-dir",
                    str(output),
                ],
            )
            reports.append(output / "pilot_b.json")
        selected_report = select_pilot_checkpoint(reports)
        selected = Path(json.loads(selected_report.read_text(encoding="utf-8"))["checkpoint"])
        self.state.setdefault("generator_pilot_selection", {})
        cast(dict[str, object], self.state["generator_pilot_selection"])[name] = {
            "checkpoint": str(selected),
            "epoch": selected.stem,
            "report": str(selected_report),
        }
        return selected

    def step0(self) -> dict[str, Any]:
        """Run and strictly validate the preregistered shift decision."""
        output = self.root / "step0"
        require_fresh(output)
        self.command(
            "step0",
            [
                "bash",
                "hpc/run.sh",
                "motif-shift-diagnostic",
                "--checkpoint",
                str(WAVE3_CHECKPOINT),
                "--data-root",
                str(self.data_root),
                "--strategy",
                "breadth_first",
                "--output-dir",
                str(output),
                "--pack-dir",
                str(self.pack_dir),
            ],
        )
        decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
        decisions = decision.get("decisions")
        if not isinstance(decisions, dict) or not all(
            isinstance(decisions.get(key), bool) for key in ("run_shift", "run_confidence", "stop")
        ):
            raise ValueError("Step 0 decision.json has no valid decisions block")
        if decisions["stop"] != (not decisions["run_shift"] and not decisions["run_confidence"]):
            raise ValueError("Step 0 decisions are internally inconsistent")
        return cast(dict[str, Any], decision)

    def prepare_cache(self, kind: str, checkpoints: tuple[Path, Path], *, run_seed: int) -> Path:
        """Materialize the exact corpus and predictions for one Stage-I run seed."""
        seed_root = OUTPUT_ROOT / f"motif_crossfit/seed42_k2/run_seed{run_seed}"
        manifest = seed_root / "manifest"
        if not manifest.exists():
            self.command(
                f"crossfit_manifest_seed{run_seed}",
                [
                    "bash",
                    "hpc/run.sh",
                    "motif-crossfit",
                    "prepare-manifest",
                    "--config",
                    str(CONFIG_ROOT / f"{LANES['shift']}.yaml"),
                    "--output-dir",
                    str(manifest),
                    "--run-seed",
                    str(run_seed),
                    "--world-size",
                    str(visible_gpu_count()),
                ],
            )
        for required in ("pairs.json", "nodes.json", "forbidden_nodes.json", "manifest.json"):
            if not (manifest / required).exists():
                raise RuntimeError(f"cross-fit manifest is missing {required}: {manifest}")
        cache = seed_root / kind / "cache.pt"
        require_fresh(cache)
        self.command(
            f"cache_{kind}_seed{run_seed}",
            [
                "bash",
                "hpc/run.sh",
                "motif-crossfit",
                "prepare",
                "--pairs",
                str(manifest / "pairs.json"),
                "--nodes",
                str(manifest / "nodes.json"),
                "--forbidden-nodes",
                str(manifest / "forbidden_nodes.json"),
                "--fold0-checkpoint",
                str(checkpoints[0]),
                "--fold1-checkpoint",
                str(checkpoints[1]),
                "--deployed-checkpoint",
                str(self.deployed_generators[kind]),
                "--pack-dir",
                str(self.pack_dir),
                "--data-root",
                str(self.data_root),
                "--output",
                str(cache),
            ],
        )
        if not cache.exists():
            raise RuntimeError(f"cross-fit preparation exited without cache: {cache}")
        if not cache.with_suffix(cache.suffix + ".report.json").exists():
            raise RuntimeError(f"cross-fit preparation exited without comparison report: {cache}")
        return cache

    def prepare_crossfit(self, kind: str, names: Sequence[str]) -> None:
        """Train a generator family and materialize its seed-0 cache."""
        selected = [self.select_generator(name, self.train(name)) for name in names]
        checkpoints = (selected[-2], selected[-1])
        if kind == "presence":
            self.deployed_presence_checkpoint = selected[0]
            self.deployed_generators["presence"] = selected[0]
        self.generator_runs[kind] = checkpoints
        self.caches[kind] = self.prepare_cache(kind, checkpoints, run_seed=0)

    def train_lane(self, lane: str) -> Path:
        """Train a lane, binding confidence to the pilot-selected deployed generator."""
        name = LANES[lane]
        source = CONFIG_ROOT / f"{name}.yaml"
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        motif = cast(dict[str, Any], payload["model"]["config"]["motif_prompt"])
        if lane == "confidence":
            if self.deployed_presence_checkpoint is None:
                raise RuntimeError(
                    "confidence lane has no pilot-selected deployed presence generator"
                )
            motif["deployed_generator_checkpoint"] = str(self.deployed_presence_checkpoint)
            motif["corruption"]["predicted_cache_path"] = str(self.caches["presence"])
        else:
            motif["deployed_generator_checkpoint"] = str(WAVE3_CHECKPOINT)
            if lane == "shift":
                motif["corruption"]["predicted_cache_path"] = str(self.caches["flat"])
        generated = self.root / "configs" / source.name
        generated.parent.mkdir(parents=True, exist_ok=True)
        require_fresh(generated)
        generated.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        self.lane_configs[name] = generated
        return self.train_config(name, generated)

    def _score(
        self, checkpoint: Path, output: Path, universe: str, intervention: str = "none"
    ) -> None:
        require_fresh(output)
        command = [
            "bash",
            "hpc/run.sh",
            "score",
            "--checkpoint",
            str(checkpoint),
            "--pairs",
            universe,
            "--data-root",
            str(self.data_root),
            "--strategy",
            "breadth_first",
            "--pack-dir",
            str(self.pack_dir),
            "--output",
            str(output),
        ]
        if intervention != "none":
            command.extend(
                ["--prefix-intervention", intervention, "--prefix-intervention-seed", "0"]
            )
        self.command(f"score_{output.parent.parent.name}_{output.stem}", command)

    def validate_lane(self, lane: str, run: Path) -> dict[str, object]:
        """Score one lane and its preregistered V_val controls."""
        analysis = self.root / "validation" / lane
        require_fresh(analysis)
        scores = analysis / "scores"
        scores.mkdir(parents=True)
        cases = {"none": "none", **{name: name for name in INTERVENTIONS}}
        for case, intervention in cases.items():
            for universe in ("val_cls", "val_topology"):
                self._score(
                    run / "best.pt", scores / f"{case}_{universe}.npz", universe, intervention
                )
        for universe in ("val_cls", "val_topology"):
            self._score(
                PREFIX_BASE_RUN / "best.pt", scores / f"prefix_base_{universe}.npz", universe
            )
        report = analyze_validation(scores, self.data_root, list(cases) + ["prefix_base"])
        atomic_json(analysis / "report.json", report)
        return report

    def test_and_controls(self, lane: str, run: Path) -> None:
        """Open held-out test once, then run density and five interventions."""
        require_fresh(run / "test_report.json")
        self.command(
            "test_winner",
            [
                "bash",
                "hpc/run.sh",
                "test",
                "--checkpoint",
                str(run / "best.pt"),
                "--output-dir",
                str(run),
                "--data-root",
                str(self.data_root),
                "--strategy",
                "breadth_first",
                "--arm",
                lane,
                "--seed",
                "0",
            ],
        )
        if not (run / "test_report.json").exists():
            raise RuntimeError("winner test exited without test_report.json")
        density = self.root / "winner_density.json"
        require_fresh(density)
        self.command(
            "winner_density",
            [
                sys.executable,
                "-m",
                "src.experiments.motif_density_control",
                "--arm",
                str(run),
                "--reference",
                str(PREFIX_BASE_RUN),
                "--output",
                str(density),
                "--data-root",
                str(self.data_root),
                "--strategy",
                "breadth_first",
            ],
        )
        intervention_root = self.root / "winner_test_interventions"
        intervention_root.mkdir()
        for intervention in FINAL_INTERVENTIONS:
            for universe in ("test", "test_topology"):
                output = intervention_root / f"{intervention}_{universe}.npz"
                self._score(run / "best.pt", output, universe, intervention)
        report = analyze_test_interventions(
            intervention_root,
            run / "test_report.json",
            self.data_root,
            list(FINAL_INTERVENTIONS),
        )
        atomic_json(intervention_root / "report.json", report)

    def run(self) -> None:
        """Execute the conditional campaign to a terminal queue status."""
        try:
            self.wait()
            decision = self.step0()
            decisions = cast(dict[str, bool], decision["decisions"])
            self.state["step0_decision"] = decisions
            if decisions["stop"]:
                self.status("stopped_negative", stage="step0", decision=decision)
                return
            if decisions["run_shift"]:
                self.prepare_crossfit("flat", FLAT_FOLDS)
            if decisions["run_confidence"]:
                self.prepare_crossfit("presence", PRESENCE_GENERATORS)
            lane_names = ["control"]
            if decisions["run_shift"]:
                lane_names.insert(0, "shift")
            if decisions["run_confidence"]:
                lane_names.insert(1 if decisions["run_shift"] else 0, "confidence")
            lane_runs: dict[str, Path] = {}
            reports: dict[str, dict[str, object]] = {}
            for lane in lane_names:
                lane_runs[lane] = self.train_lane(lane)
                reports[lane] = self.validate_lane(lane, lane_runs[lane])
            winner = select_lane(lane_runs)
            adopted = adoption(reports[winner])
            self.state.update(winner=winner, validation=reports, adoption=adopted)
            if not adopted["adopt"]:
                self.status("stopped_negative", stage="v_val_decision")
                return
            for seed in (1, 2):
                kind = {"shift": "flat", "confidence": "presence"}.get(winner)
                cache = (
                    None
                    if kind is None
                    else self.prepare_cache(kind, self.generator_runs[kind], run_seed=seed)
                )
                replica = self.replicate(LANES[winner], seed, cache)
                reports[f"{winner}_seed{seed}"] = self.validate_lane(
                    f"{winner}_seed{seed}", replica
                )
            self.test_and_controls(LANES[winner], lane_runs[winner])
            self.status("complete", stage="winner_test_and_controls")
        except NoPassingPilotError as error:
            self.status(
                "stopped_negative",
                stage="generator_pilot_b",
                error_type=type(error).__name__,
                error=str(error),
                reports=[str(path) for path in error.reports],
            )
        except Exception as error:
            self.status(
                "failed",
                stage=str(self.state.get("stage", "unknown")),
                error_type=type(error).__name__,
                error=str(error),
            )
            raise


def _auprc(path: Path) -> float:
    artifact = load_scores(path)
    validate_artifact_precision(artifact, label=str(path))
    return float(average_precision_score(artifact.label, artifact.logit))


def analyze_validation(scores: Path, data_root: Path, cases: list[str]) -> dict[str, object]:
    """Read V_val interventions at protocol and matched-density operating points."""
    split = _load_val_region_split(data_root, "breadth_first")
    graph = split.build_g_val()
    config = MMDConfig()
    artifacts = {case: load_scores(scores / f"{case}_val_topology.npz") for case in cases}
    for case, artifact in artifacts.items():
        validate_artifact_precision(artifact, label=f"{case} val_topology")
    pairs = list(artifacts["none"].pairs())
    if any(list(artifact.pairs()) != pairs for artifact in artifacts.values()):
        raise ValueError("V_val topology artifacts do not share one ordered universe")
    selection = select_fixed_threshold(
        pairs=pairs,
        logits=artifacts["none"].logit.astype(np.float64),
        g_ref=graph,
        buckets=split.buckets,
        config=config,
    )
    selected = cast(dict[str, object], selection.report["selected"])
    threshold = cast(float, selected["logit_threshold"])
    topology = {
        case: evaluate_fixed_threshold(
            pairs=pairs,
            logits=artifact.logit.astype(np.float64),
            g_ref=graph,
            buckets=split.buckets,
            threshold=threshold,
            config=config,
        )[1]
        for case, artifact in artifacts.items()
    }
    target_edges = cast(int, selected["union_admitted_pair_count"])
    matched = density_matched_report(
        rows={case: artifact.logit.astype(np.float64) for case, artifact in artifacts.items()},
        union_pairs=pairs,
        target_edges=target_edges,
        g_ref=graph,
        buckets=split.buckets,
        config=config,
    )
    return {
        "scope": "V_val_only",
        "cases": {
            case: {
                "auprc": _auprc(scores / f"{case}_val_cls.npz"),
                "topology_at_lane_threshold": topology[case],
            }
            for case in cases
        },
        "lane_threshold_selection": selection.report,
        "matched_output_density": matched,
    }


def analyze_test_interventions(
    scores: Path, report_path: Path, data_root: Path, cases: list[str]
) -> dict[str, object]:
    """Report every held-out intervention at the winner's frozen V_val thresholds."""
    published = json.loads(report_path.read_text(encoding="utf-8"))
    cls_threshold = float(published["classification_threshold"]["logit_threshold"])
    topology_threshold = float(
        published["graph"]["fixed_threshold"]["validation_selection"]["selected"]["logit_threshold"]
    )
    benchmark_root = data_root / _BENCHMARK_SUBDIR
    graph = load_test_graph(benchmark_root, "breadth_first")
    buckets = load_test_node_buckets(benchmark_root, "breadth_first")
    output: dict[str, object] = {}
    for case in cases:
        topology_path = scores / f"{case}_test_topology.npz"
        artifact = load_scores(topology_path)
        validate_artifact_precision(artifact, label=str(topology_path))
        _, topology = evaluate_fixed_threshold(
            pairs=list(artifact.pairs()),
            logits=artifact.logit.astype(np.float64),
            g_ref=graph,
            buckets=buckets,
            threshold=topology_threshold,
            config=MMDConfig(),
        )
        output[case] = {
            "edge": report_edge_metrics(
                scores / f"{case}_test.npz",
                expect_pairs_source="test",
                logit_threshold=cls_threshold,
            ),
            "topology": topology,
        }
    return {
        "scope": "held_out_test",
        "classification_logit_threshold_from_v_val": cls_threshold,
        "topology_logit_threshold_from_v_val": topology_threshold,
        "cases": output,
    }


def select_lane(runs: dict[str, Path]) -> str:
    """Select the lane with the production five-metric mean-rank rule."""
    names = list(runs)
    metrics = [read_run(runs[name]) for name in names]
    selected = select_checkpoint(
        [
            CheckpointCandidate(index + 1, item.auprc, item.topology)
            for index, item in enumerate(metrics)
        ]
    )
    if selected is None:
        raise ValueError("no completed v17 lanes")
    return names[selected.epoch - 1]


def adoption(report: dict[str, object]) -> dict[str, object]:
    """Apply the preregistered +0.004 AUPRC and matched-density GS rule."""
    cases = cast(dict[str, dict[str, object]], report["cases"])
    arm_auprc = cast(float, cases["none"]["auprc"])
    gates_auprc = cast(float, cases["gates_off"]["auprc"])
    gain = arm_auprc - gates_auprc
    clears_gain = Decimal(str(arm_auprc)) - Decimal(str(gates_auprc)) > Decimal("0.004")
    matched = cast(dict[str, object], report["matched_output_density"])
    rows = cast(dict[str, dict[str, object]], matched["rows"])
    arm_panel = cast(dict[str, Any], rows["none"]["panel"])
    base_panel = cast(dict[str, Any], rows["prefix_base"]["panel"])
    arm_gs = float(cast(dict[str, float], arm_panel["graph_similarity"])["bfs_macro"])
    base_gs = float(cast(dict[str, float], base_panel["graph_similarity"])["bfs_macro"])
    return {
        "adopt": clears_gain and arm_gs >= base_gs,
        "auprc_gain_over_gates_off": gain,
        "required_gain_exclusive": 0.004,
        "matched_density_gs": arm_gs,
        "prefix_base_matched_density_gs": base_gs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--pack-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    predecessor = process_identity(args.wait_pid)
    if predecessor is None:
        raise RuntimeError(f"--wait-pid {args.wait_pid} is not a live process")
    args.output.mkdir(parents=True, exist_ok=False)
    lock_path = args.output / ".lock"
    with lock_path.open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Queue(
            root=args.output,
            predecessor=predecessor,
            data_root=args.data_root,
            pack_dir=args.pack_dir,
        ).run()


if __name__ == "__main__":
    main()


__all__ = [
    "Queue",
    "adoption",
    "analyze_validation",
    "process_identity",
    "same_process",
    "select_lane",
]
