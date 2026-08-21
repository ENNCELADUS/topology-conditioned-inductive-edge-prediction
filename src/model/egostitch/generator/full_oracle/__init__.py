"""Full-neighborhood ground-truth oracle generator and its features-only twin."""

from .features import FeatureEgoGraph, FullEgoFeaturesGenerator
from .generator import FullEgoGraph, FullOracleGenerator

__all__ = ["FeatureEgoGraph", "FullEgoFeaturesGenerator", "FullEgoGraph", "FullOracleGenerator"]
