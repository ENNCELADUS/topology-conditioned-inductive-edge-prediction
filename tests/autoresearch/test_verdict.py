import json
from pathlib import Path

import pytest
from src.autoresearch.judge import main as judge_main
from src.autoresearch.metrics_io import RunMetrics
from src.autoresearch.verdict import judge_runs, undominated
from src.eval.checkpoint_selection import TopologyValidationMetrics

from tests.autoresearch.conftest import RunDirFactory, make_metric_row

pytestmark = pytest.mark.unit


def run_with(
    gs: float = 0.50,
    rd: float = 1.10,
    degree_mmd: float = 0.90,
    clustering_mmd: float = 0.85,
    spectral_mmd: float = 0.80,
    auprc: float = 0.82,
) -> RunMetrics:
    return RunMetrics(
        run_dir=Path("fixture"),
        selected_epoch=1,
        auprc=auprc,
        topology=TopologyValidationMetrics(
            gs=gs,
            rd=rd,
            degree_mmd=degree_mmd,
            clustering_mmd=clustering_mmd,
            spectral_mmd=spectral_mmd,
        ),
        threshold=2.5,
        total_seconds=1.0,
    )


def test_keep_when_one_metric_strictly_improves() -> None:
    verdict = judge_runs(run_with(), run_with(gs=0.60))
    assert verdict.decision == "keep"
    assert verdict.improved == ("gs",)
    assert verdict.regressed == ()
    assert verdict.reasons == ("improved without regression: gs",)
    assert verdict.deltas["gs"] == pytest.approx(-0.10)


def test_revert_on_exact_tie() -> None:
    verdict = judge_runs(run_with(), run_with())
    assert verdict.decision == "revert"
    assert verdict.improved == ()
    assert verdict.regressed == ()
    assert verdict.reasons == ("no topology metric improved beyond tolerance",)


def test_revert_when_any_metric_regresses() -> None:
    verdict = judge_runs(run_with(), run_with(gs=0.60, degree_mmd=0.95))
    assert verdict.decision == "revert"
    assert verdict.improved == ("gs",)
    assert verdict.regressed == ("degree_mmd",)
    assert verdict.reasons == ("regressed beyond tolerance: degree_mmd",)


def test_rd_is_judged_by_absolute_log_distance_from_one() -> None:
    verdict = judge_runs(run_with(rd=1.10), run_with(rd=0.90))
    assert verdict.regressed == ("log_rd",)
    assert verdict.decision == "revert"


def test_bands_absorb_small_moves_in_both_directions() -> None:
    bands = {"gs": 0.02, "degree_mmd": 0.02}
    small_both_ways = run_with(gs=0.51, degree_mmd=0.91, spectral_mmd=0.70)
    verdict = judge_runs(run_with(), small_both_ways, bands)
    assert verdict.improved == ("spectral_mmd",)
    assert verdict.regressed == ()
    assert verdict.decision == "keep"


def test_auprc_never_enters_the_decision() -> None:
    verdict = judge_runs(run_with(auprc=0.82), run_with(gs=0.60, auprc=0.10))
    assert verdict.decision == "keep"
    assert verdict.auprc_delta == pytest.approx(-0.72)


@pytest.mark.parametrize(
    "bands",
    [
        {"unknown": 0.1},
        {"gs": True},
        {"gs": "0.1"},
        {"gs": None},
        {"gs": float("nan")},
        {"gs": float("inf")},
        {"gs": -0.1},
    ],
)
def test_invalid_band_rejected_by_api(bands: dict[str, object]) -> None:
    regression = run_with(gs=0.60, degree_mmd=0.95)
    with pytest.raises(ValueError, match="band"):
        judge_runs(run_with(), regression, bands)  # type: ignore[arg-type]


def test_undominated_filters_pareto_dominated_runs() -> None:
    best = run_with(gs=0.60)
    dominated = run_with(gs=0.40, spectral_mmd=0.90)
    tradeoff = run_with(gs=0.40, spectral_mmd=0.10)
    survivors = undominated([best, dominated, tradeoff])
    assert survivors == [best, tradeoff]


def test_judge_cli_emits_verdict_json(
    make_run_dir: RunDirFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    incumbent = make_run_dir(name="incumbent", selected_epoch=1)
    trial_rows = [make_metric_row(1, val_gs_bfs=0.60)]
    trial = make_run_dir(name="trial", rows=trial_rows, selected_epoch=1)
    assert judge_main(["--incumbent", str(incumbent), "--trial", str(trial)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "keep"
    assert payload["improved"] == ["gs"]
    assert payload["reasons"] == ["improved without regression: gs"]
    assert payload["deltas"]["gs"] == pytest.approx(-0.09)
    assert payload["trial"]["gs"] == pytest.approx(0.60)
    assert payload["incumbent"]["auprc"] == pytest.approx(0.81)


@pytest.mark.parametrize(
    "bands",
    [
        {"unknown": 0.1},
        {"degree_mmd": True},
        {"degree_mmd": "0.1"},
        {"degree_mmd": None},
        {"degree_mmd": float("nan")},
        {"degree_mmd": float("inf")},
        {"degree_mmd": -0.1},
    ],
)
def test_invalid_band_rejected_by_cli_without_suppressing_regression(
    make_run_dir: RunDirFactory, tmp_path: Path, bands: dict[str, object]
) -> None:
    incumbent = make_run_dir(name="incumbent", selected_epoch=1)
    trial = make_run_dir(
        name="trial",
        rows=[make_metric_row(1, val_gs_bfs=0.60, val_degree_mmd_ratio=0.95)],
        selected_epoch=1,
    )
    bands_path = tmp_path / "bands.json"
    bands_path.write_text(json.dumps(bands), encoding="utf-8")
    with pytest.raises(ValueError, match="band"):
        judge_main(
            [
                "--incumbent",
                str(incumbent),
                "--trial",
                str(trial),
                "--bands",
                str(bands_path),
            ]
        )
