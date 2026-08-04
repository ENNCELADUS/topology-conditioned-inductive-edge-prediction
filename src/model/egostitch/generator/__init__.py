"""Neighborhood graph generator component (three-component refactor design §3.3, §5).

Consumes two endpoints' features and grounding pools, emits one
`src.model.egostitch.graph.ImaginedGraph` (or `None`, the null case), and
owns every auxiliary loss that supervises its own imagination.
"""

from src.model.egostitch.generator.base import NeighborhoodGenerator
from src.model.egostitch.generator.egostitch import (
    EgoStitchImagineGenerator,
    GeneratorNodeState,
    StitchedGraph,
)
from src.model.egostitch.generator.null import NullGenerator
from src.model.egostitch.generator.oracle import (
    OracleScaffoldTable,
    OracleStructGenerator,
    build_oracle_table,
)

__all__ = [
    "NeighborhoodGenerator",
    "EgoStitchImagineGenerator",
    "GeneratorNodeState",
    "StitchedGraph",
    "NullGenerator",
    "OracleScaffoldTable",
    "OracleStructGenerator",
    "build_oracle_table",
]
