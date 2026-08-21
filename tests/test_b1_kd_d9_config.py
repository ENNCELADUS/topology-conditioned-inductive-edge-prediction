"""Contract test for the runnable B1 KD-D9 configuration."""

from pathlib import Path

from src.model.egostitch.classifier.b0_v31 import V3_1
from src.score_universe import build_model
from src.train_b0 import load_config

CONFIG_PATH = Path("configs/b1_kd_d9_breadth_first.yaml")


def test_kd_d9_config_builds_pair_latent_arm() -> None:
    config = load_config(CONFIG_PATH)

    assert config.output_dir == Path("outputs/b1_row_kd/kd_d9")
    assert config.distill is not None
    assert config.distill.arm == "kd_d9"
    assert config.distill.targets_path == "outputs/distill/kd_row_targets_breadth_first"
    assert config.distill.w_seed == 1.0
    assert config.distill.w_geom == 0.3
    assert config.distill.w_kl == 0.05
    assert config.distill.kl_warmup_steps == 2000
    assert config.distill.joint_start_epoch == 13
    assert config.distill.gen_lr_scale == 0.1
    assert config.mixed_precision == "bf16"

    assert config.model.config["pair_latent_gen"] == {
        "z_dim": 64,
        "cond_dim": 256,
        "hidden": 512,
        "seed_count": 4,
        "seed_dim": 512,
        "mc_samples": 4,
        "kl_free_bits": 0.05,
    }

    model = build_model(config.model.family, config.model.config)
    assert isinstance(model, V3_1)
    assert model.pair_latent_gen is not None
    assert model.pair_latent_gen.z_dim == 64
    assert model.pair_latent_gen.seed_count == 4
    assert model.pair_latent_gen.seed_dim == 512
    assert model.pair_latent_gen.mc_samples == 4
