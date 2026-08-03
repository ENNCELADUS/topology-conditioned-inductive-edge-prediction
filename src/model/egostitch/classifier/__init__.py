"""Pairwise classifier component (three-component refactor design §3.3, §5).

Split into a cacheable per-node `encode_tokens` phase and a pair-level
`forward` phase (2026-08-03): `forward` consumes `PairInputs` (already-
encoded endpoint token states, produced by `encode_tokens`) and an optional
`PairConditioning`, emits the edge logit. `cond=None` is the unconditioned
path -- with a null generator this is exactly the pure B0 pairwise baseline.
"""

from src.model.egostitch.classifier.b0_v31 import B0V31PairClassifier
from src.model.egostitch.classifier.base import PairClassifier

__all__ = ["PairClassifier", "B0V31PairClassifier"]
