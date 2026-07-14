"""EgoStitch Stage-1 model package (spec docs/05-egostitch-spec.md Sec 13).

Stage 1 = Tokenize-lite + Imagine + Hungarian matching + Sinkhorn Stitch +
decision head over (s0, s1, s2). No codebook, no harmonization, no CVAE.
"""

from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.model import EgoStitchStage1

__all__ = ["EgoStitchConfig", "EgoStitchStage1"]
