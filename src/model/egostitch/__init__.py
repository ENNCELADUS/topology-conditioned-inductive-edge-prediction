"""EgoStitch Stage-1 model package (spec docs/05-egostitch-spec.md Sec 13).

Stage 1 = Tokenize-lite + Imagine + Hungarian matching + Sinkhorn Stitch. No
codebook, no harmonization, no CVAE. The frozen-s0 decision head over
(s0, s1, s2) was retired with `decision.py` (three-component refactor design
§9); `EgoStitchE2E` scores pairs through its own trunk instead.
"""

from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.model import EgoStitchStage1

__all__ = ["EgoStitchConfig", "EgoStitchStage1"]
