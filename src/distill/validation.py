"""G_val oracle comparisons on fixed val_cls rows; never an optimization objective."""

from collections.abc import Sequence

import numpy as np
import torch

from src.data.val_region import Pair
from src.distill.artifacts import KDRowTargets
from src.distill.config import DistillConfig
from src.distill.losses import kd_dist_loss, kd_gram_loss, kd_logit_loss, kd_rank_loss, kd_rep_loss


class OracleValidationBank:
    """Join a separate G_val bank to the exact student validation rows."""

    def __init__(
        self,
        targets: KDRowTargets,
        cfg: DistillConfig,
        pairs: Sequence[Pair],
        labels: Sequence[int],
    ) -> None:
        if targets.manifest.get("truth_source") != "validation_structure":
            raise ValueError("validation diagnostics require teacher scores from G_val")
        actual = [
            (targets.node_ids[a], targets.node_ids[b])
            for a, b in zip(targets.pair_a_idx, targets.pair_b_idx, strict=True)
        ]
        if actual != list(pairs) or not np.array_equal(targets.pair_label, labels):
            raise ValueError("oracle validation bank pair/label join mismatch")
        if (
            not np.isfinite(targets.teacher_logit).all()
            or not np.isfinite(targets.teacher_rep).all()
        ):
            raise ValueError("oracle validation targets must be finite")
        self.targets = targets
        self.cfg = cfg
        self.rep_key = "kd_rep" if cfg.w_rep > 0 else "pair_repr" if cfg.w_gram > 0 else None

    @torch.no_grad()
    def metrics(self, logits: np.ndarray, rep: np.ndarray | None) -> dict[str, float]:
        """Unweighted losses; fixed row blocks/anchor groups independent of DDP batching."""
        student = torch.as_tensor(logits, dtype=torch.float32)
        teacher = torch.from_numpy(self.targets.teacher_logit)
        if student.shape != teacher.shape or not torch.isfinite(student).all():
            raise ValueError("invalid student validation logits")
        out = {
            "val_kd_logit_loss": float(kd_logit_loss(student, teacher)),
            "val_prob_mae": float((student.sigmoid() - teacher.sigmoid()).abs().mean()),
        }
        if student.numel() > 1 and student.std() > 0 and teacher.std() > 0:
            out["val_logit_pearson"] = float(torch.corrcoef(torch.stack((student, teacher)))[0, 1])
        if self.cfg.w_rank > 0:
            # Each undirected pair participates at both endpoints; self-pairs only once.
            a, b = self.targets.pair_a_idx, self.targets.pair_b_idx
            rows = np.concatenate((np.arange(len(a)), np.flatnonzero(a != b)))
            groups = torch.from_numpy(np.concatenate((a, b[a != b])).astype(np.int64))
            out["val_kd_rank_loss"] = float(
                kd_rank_loss(student[rows], teacher[rows], groups, margin=self.cfg.margin)
            )
            out["val_kd_dist_loss"] = float(kd_dist_loss(student[rows], teacher[rows], groups))
        if self.rep_key is not None:
            if rep is None or len(rep) != len(student) or not np.isfinite(rep).all():
                raise ValueError("invalid student validation representations")
            s_rep = torch.as_tensor(rep, dtype=torch.float32)
            t_rep = torch.as_tensor(self.targets.teacher_rep, dtype=torch.float32)
            if self.cfg.w_rep > 0:
                out["val_kd_rep_loss"] = float(kd_rep_loss(s_rep, t_rep))
                out["val_rep_cos"] = 1.0 - out["val_kd_rep_loss"]
            if self.cfg.w_gram > 0:
                # Bound quadratic memory with deterministic 256-row blocks.
                total, weight = 0.0, 0
                for start in range(0, len(student), 256):
                    stop = min(start + 256, len(student))
                    n = (stop - start) * (stop - start - 1)
                    if n:
                        total += n * float(kd_gram_loss(s_rep[start:stop], t_rep[start:stop]))
                        weight += n
                out["val_kd_gram_block_loss"] = total / weight if weight else 0.0
        return out
