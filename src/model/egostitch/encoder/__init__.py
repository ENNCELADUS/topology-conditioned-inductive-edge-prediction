"""Graph encoder component (three-component refactor design §3.3, §5).

Consumes an `ImaginedGraph`, emits a `GraphEmbedding`, and owns the
train-only relational auxiliary loss (`L_rel`) that supervises its own
representation.
"""

from src.model.egostitch.encoder.base import GraphEncoder
from src.model.egostitch.encoder.grit_gmt import GritGmtEncoder
from src.model.egostitch.encoder.ste import TypedMessagePassingEncoder

__all__ = ["GraphEncoder", "GritGmtEncoder", "TypedMessagePassingEncoder"]
