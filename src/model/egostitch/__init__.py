"""EgoStitch Stage-1 model package (spec docs/05-egostitch-spec.md Sec 13).

Stage 1 = Tokenize-lite + Imagine + Hungarian matching + Sinkhorn Stitch. No
codebook, no harmonization, no CVAE. The frozen-s0 decision head over
(s0, s1, s2) was retired with `decision.py` (three-component refactor design
§9). `EgoStitchModel` (`composite.py`) replaces the monolithic `EgoStitchE2E`
(three-component refactor design §5): it wires the `generator`
(`NeighborhoodGenerator`), `encoder` (`GraphEncoder`) and `classifier`
(`PairClassifier`) components together and scores pairs through
`classifier`'s trunk.
"""

from src.model.egostitch.composite import E2ENodeState, E2EPairContext, EgoStitchModel
from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.generator.imagine import EgoStitchStage1

__all__ = [
    "E2ENodeState",
    "E2EPairContext",
    "EgoStitchConfig",
    "EgoStitchModel",
    "EgoStitchStage1",
]
