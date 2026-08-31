"""kd_gen topology-latent generator families: edm | imf | det_mse.

One 512-D latent target (the PMA(1) teacher's pooled seed, unit-RMS
normalized by `latent_rms_scale`), one shared conditional residual-MLP core,
one fusion path (`marginal_forward`) shared by `V3_1.forward` and the packed
scorer. The whole module is an fp32 island; probability-space Monte-Carlo
marginalization emits `logit(mean_m sigmoid(l_m))`. Spec:
docs/superpowers/specs/2026-08-30-kd-gen-arm-design.md.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import torch
from torch import nn

# Noise/sampler constants relative to unit data RMS (sigma_data == 1).
P_MEAN = -0.51
P_STD = 1.2
SIGMA_MIN = 0.004
SIGMA_MAX = 160.0
RHO = 7.0
_NOISE_EMBED_DIM = 64
_EPS_SEED = 20260830
# Quartile edges of ln sigma ~ N(P_MEAN, P_STD^2): P_MEAN +/- 0.6745 * P_STD.
_SIGMA_QUARTILES = (
    math.exp(P_MEAN - 0.6745 * P_STD),
    math.exp(P_MEAN),
    math.exp(P_MEAN + 0.6745 * P_STD),
)
FAMILIES = ("edm", "imf", "det_mse")
CONTROLS = ("branch_zero", "shuffle")


def marginal_logit(per_sample: torch.Tensor) -> torch.Tensor:
    """``logit(mean_m sigmoid(l_m))`` computed stably: (B, M) -> (B,)."""
    log_m = math.log(per_sample.size(1))
    log_p = torch.logsumexp(-nn.functional.softplus(-per_sample), dim=1) - log_m
    log_q = torch.logsumexp(-nn.functional.softplus(per_sample), dim=1) - log_m
    return log_p - log_q


def _masked_token_mean(tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(tokens.size(1), device=tokens.device)
    valid = (positions.unsqueeze(0) < lengths.unsqueeze(1)).to(dtype=tokens.dtype)
    total = (tokens * valid.unsqueeze(-1)).sum(dim=1)
    return total / valid.sum(dim=1).clamp_min(1.0).unsqueeze(-1)


def _fourier_embed(values: torch.Tensor, dim: int = _NOISE_EMBED_DIM) -> torch.Tensor:
    """Sinusoidal embedding of a per-row scalar: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=values.device, dtype=values.dtype)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    args = values.unsqueeze(-1) * freqs
    return torch.cat((torch.sin(args), torch.cos(args)), dim=-1)


class _AdaLNBlock(nn.Module):
    """AdaLN-Zero residual MLP block: LN -> (1+scale)*x+shift -> MLP -> gate."""

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mod = nn.Linear(cond_dim, 3 * dim)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.mod(cond_vec).chunk(3, dim=-1)
        return cast(
            torch.Tensor,
            x + gate * cast(torch.Tensor, self.mlp(self.norm(x) * (1.0 + scale) + shift)),
        )


class _ResidualCore(nn.Module):
    """The shared denoiser body F(x; cond_vec): R^latent -> R^latent."""

    def __init__(self, latent_dim: int, cond_dim: int, blocks: int) -> None:
        super().__init__()
        self.cond_in = nn.Linear(cond_dim, latent_dim)
        self.blocks = nn.ModuleList(_AdaLNBlock(latent_dim, cond_dim) for _ in range(blocks))
        self.out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        x = x + self.cond_in(cond_vec)
        for block in self.blocks:
            x = block(x, cond_vec)
        return cast(torch.Tensor, self.out(x))


class TopoGenBase(nn.Module):
    """Shared conditioning, fusion adapter, sampling scaffold, marginalization."""

    family = "base"

    def __init__(
        self,
        d_model: int,
        *,
        latent_dim: int,
        cond_dim: int,
        blocks: int,
        adapter_dim: int,
        mc_samples: int,
        sampler_steps: int,
    ) -> None:
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("cond_dim", cond_dim),
            ("blocks", blocks),
            ("adapter_dim", adapter_dim),
            ("mc_samples", mc_samples),
            ("sampler_steps", sampler_steps),
        ):
            if value <= 0:
                raise ValueError(f"topo_gen.{name} must be positive, got {value}")
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.adapter_dim = adapter_dim
        self.mc_samples = mc_samples
        self.sampler_steps = sampler_steps
        self.joint_stage = False
        self.control: str | None = None
        self.condition_net = nn.Sequential(
            nn.Linear(3 * d_model, cond_dim), nn.GELU(), nn.LayerNorm(cond_dim)
        )
        self.core = _ResidualCore(latent_dim, cond_dim, blocks)
        self.cond_embed = nn.Sequential(nn.Linear(cond_dim + _NOISE_EMBED_DIM, cond_dim), nn.GELU())
        # Fusion: pooled residual adapter. Exactly ONE zero factor (spec §6):
        # W_up zero-init carries the init identity; the gate starts at 1.0 —
        # both at zero is a permanent saddle (both gradients vanish).
        self.adapter_down = nn.Linear(latent_dim, adapter_dim)
        self.adapter_up = nn.Linear(adapter_dim, d_model)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.gate = nn.Parameter(torch.ones(()))
        self.register_buffer("latent_rms_scale", torch.ones(()))
        eps_generator = torch.Generator().manual_seed(_EPS_SEED)
        self.register_buffer(
            "eval_eps", torch.randn(mc_samples, latent_dim, generator=eps_generator)
        )

    def set_rms_scale(self, scale: float) -> None:
        """Stamp the artifact-derived normalization scalar (checkpointed buffer)."""
        if not scale > 0.0:
            raise ValueError(f"latent_rms_scale must be positive, got {scale}")
        cast(torch.Tensor, self.latent_rms_scale).fill_(scale)

    def condition(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric fp32 conditioning from *detached* trunk encodings (D4 boundary)."""
        e_u = _masked_token_mean(encoded_a.detach().float(), lengths_a)
        e_v = _masked_token_mean(encoded_b.detach().float(), lengths_b)
        features = torch.cat((e_u * e_v, e_u + e_v, (e_u - e_v).abs()), dim=-1)
        return cast(torch.Tensor, self.condition_net(features))

    def _noise_draws(self, batch: int, device: torch.device) -> torch.Tensor:
        if self.training:
            return torch.randn(batch, self.mc_samples, self.latent_dim, device=device)
        eval_eps = cast(torch.Tensor, self.eval_eps)
        return eval_eps.to(device=device).unsqueeze(0).expand(batch, -1, -1)

    def _sample_from_eps(self, cond: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_latents(self, cond: torch.Tensor) -> torch.Tensor:
        """(B, cond_dim) -> (B, M, latent_dim) normalized-space samples."""
        eps = self._noise_draws(cond.size(0), cond.device)
        return self._sample_from_eps(cond, eps)

    def gen_loss(
        self, cond: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the family-specific generator loss and telemetry."""
        raise NotImplementedError

    def generator_parameters(self) -> list[nn.Parameter]:
        """Condition + core + noise embed — the warmup/joint LR param group."""
        modules: tuple[nn.Module, ...] = (self.condition_net, self.core, self.cond_embed)
        return [p for module in modules for p in module.parameters()]

    def marginal_forward(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        pair_repr: torch.Tensor,
        output_head: nn.Module,
    ) -> dict[str, torch.Tensor]:
        """The deployed path, shared by `V3_1.forward` and the packed scorer."""
        with torch.autocast(device_type=pair_repr.device.type, enabled=False):
            cond = self.condition(encoded_a, encoded_b, lengths_a, lengths_b)
            latents = self.sample_latents(cond)  # (B, M, D) normalized
            if self.control == "shuffle":
                latents = torch.roll(latents, shifts=1, dims=0)
            if not self.joint_stage:
                latents = latents.detach()
            z = pair_repr.float()
            delta = self.adapter_up(
                nn.functional.gelu(self.adapter_down(latents))
            )  # (B, M, d_model)
            if self.control == "branch_zero":
                delta = torch.zeros_like(delta)
            fused = z.unsqueeze(1) + torch.tanh(self.gate) * delta
            per_sample = cast(
                torch.Tensor,
                output_head(fused.reshape(-1, z.size(-1))).reshape(z.size(0), latents.size(1)),
            )
            logits = marginal_logit(per_sample).unsqueeze(-1)
            probs = torch.sigmoid(per_sample)
            if latents.size(1) > 1:
                diffs = latents.unsqueeze(1) - latents.unsqueeze(2)  # (B, M, M, D)
                pair_dist = diffs.norm(dim=-1)
                m = latents.size(1)
                dispersion = pair_dist.sum(dim=(1, 2)) / (m * (m - 1)) / math.sqrt(self.latent_dim)
            else:
                dispersion = latents.new_zeros(latents.size(0))
        return {
            "logits": logits,
            "gen_prob_std": probs.std(dim=1, unbiased=False),
            "gen_branch_ratio": (
                (torch.tanh(self.gate) * delta).norm(dim=-1).mean(dim=1)
                / z.norm(dim=-1).clamp_min(1e-8)
            ),
            "gen_sample_dispersion": dispersion,
            "gen_latent_sample": latents[:, 0],
            "gen_cond": cond,
        }


class EdmTopoGen(TopoGenBase):
    """EDM-preconditioned denoiser at sigma_data == 1, deterministic Heun sampler."""

    def _denoise(
        self, x_sigma: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        c_skip = 1.0 / (sigma.square() + 1.0)
        c_out = sigma / torch.sqrt(sigma.square() + 1.0)
        c_in = 1.0 / torch.sqrt(sigma.square() + 1.0)
        cond_vec = self.cond_embed(
            torch.cat((cond, _fourier_embed(0.25 * torch.log(sigma))), dim=-1)
        )
        raw = cast(torch.Tensor, self.core(c_in.unsqueeze(-1) * x_sigma, cond_vec))
        return c_skip.unsqueeze(-1) * x_sigma + c_out.unsqueeze(-1) * raw

    def _sigma_grid(self, device: torch.device) -> torch.Tensor:
        steps = torch.arange(self.sampler_steps, dtype=torch.float32, device=device)
        grid = (
            SIGMA_MAX ** (1.0 / RHO)
            + steps
            / max(self.sampler_steps - 1, 1)
            * (SIGMA_MIN ** (1.0 / RHO) - SIGMA_MAX ** (1.0 / RHO))
        ) ** RHO
        return torch.cat((grid, grid.new_zeros(1)))

    def _sample_from_eps(self, cond: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        batch, samples, dim = eps.shape
        flat_cond = cond.unsqueeze(1).expand(-1, samples, -1).reshape(-1, cond.size(-1))
        sigmas = self._sigma_grid(cond.device)
        x = eps.reshape(-1, dim) * sigmas[0]
        for index in range(self.sampler_steps):
            sigma_now = sigmas[index].expand(x.size(0))
            sigma_next = sigmas[index + 1].expand(x.size(0))
            d_now = (x - self._denoise(x, sigma_now, flat_cond)) / sigma_now.unsqueeze(-1)
            x_next = x + (sigma_next - sigma_now).unsqueeze(-1) * d_now
            if float(sigmas[index + 1]) > 0.0:
                d_next = (x_next - self._denoise(x_next, sigma_next, flat_cond)) / (
                    sigma_next.unsqueeze(-1)
                )
                x = x + (sigma_next - sigma_now).unsqueeze(-1) * 0.5 * (d_now + d_next)
            else:
                x = x_next
        return x.reshape(batch, samples, dim)

    def gen_loss(
        self, cond: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the EDM denoising loss and per-noise-bin telemetry."""
        with torch.autocast(device_type=cond.device.type, enabled=False):
            target = target.float()
            sigma = torch.exp(P_MEAN + P_STD * torch.randn(target.size(0), device=target.device))
            noised = target + sigma.unsqueeze(-1) * torch.randn_like(target)
            denoised = self._denoise(noised, sigma, cond)
            weight = (sigma.square() + 1.0) / sigma.square()
            per_row = weight * (denoised - target).square().mean(dim=-1)
            loss = per_row.mean()
            edges = torch.tensor(_SIGMA_QUARTILES, device=sigma.device)
            bins = torch.bucketize(sigma, edges)
            bin_loss = torch.zeros(4, device=sigma.device).scatter_add_(0, bins, per_row.detach())
            bin_count = torch.zeros(4, device=sigma.device).scatter_add_(
                0, bins, torch.ones_like(sigma)
            )
        return loss, {"sigma_bin_loss": bin_loss, "sigma_bin_count": bin_count}


class DetMseTopoGen(TopoGenBase):
    """Plain-MSE conditional-mean control: h_hat = f(c), M forced to 1 (spec §5)."""

    family = "det_mse"

    def __init__(self, d_model: int, **kwargs: int) -> None:
        kwargs["mc_samples"] = 1
        super().__init__(d_model, **kwargs)

    def _predict(self, cond: torch.Tensor) -> torch.Tensor:
        cond_vec = self.cond_embed(
            torch.cat(
                (cond, torch.zeros(cond.size(0), _NOISE_EMBED_DIM, device=cond.device)),
                dim=-1,
            )
        )
        return cast(
            torch.Tensor,
            self.core(torch.zeros(cond.size(0), self.latent_dim, device=cond.device), cond_vec),
        )

    def _sample_from_eps(self, cond: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return self._predict(cond).unsqueeze(1)

    def gen_loss(
        self, cond: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the deterministic conditional-mean MSE loss."""
        with torch.autocast(device_type=cond.device.type, enabled=False):
            loss = (self._predict(cond) - target.float()).square().mean()
        return loss, {}


_FAMILY_CLASSES: dict[str, type[TopoGenBase]] = {
    "edm": EdmTopoGen,
    "det_mse": DetMseTopoGen,
}

_KNOWN_KEYS = {
    "name",
    "latent_dim",
    "cond_dim",
    "blocks",
    "adapter_dim",
    "mc_samples",
    "sampler_steps",
}


def build_topo_gen(cfg: Mapping[str, object], d_model: int) -> TopoGenBase:
    """Build the configured generator family; strict keys, strict name."""
    unknown = sorted(set(cfg) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(f"unknown topo_gen keys: {unknown}")
    name = str(cfg.get("name", ""))
    if name not in _FAMILY_CLASSES:
        raise ValueError(f"topo_gen.name must be one of {sorted(_FAMILY_CLASSES)}, got {name!r}")
    return _FAMILY_CLASSES[name](
        d_model,
        latent_dim=int(cast(int, cfg.get("latent_dim", 512))),
        cond_dim=int(cast(int, cfg.get("cond_dim", 256))),
        blocks=int(cast(int, cfg.get("blocks", 4))),
        adapter_dim=int(cast(int, cfg.get("adapter_dim", 128))),
        mc_samples=int(cast(int, cfg.get("mc_samples", 4))),
        sampler_steps=int(cast(int, cfg.get("sampler_steps", 4))),
    )


__all__ = [
    "CONTROLS",
    "FAMILIES",
    "DetMseTopoGen",
    "EdmTopoGen",
    "TopoGenBase",
    "build_topo_gen",
    "marginal_logit",
]
