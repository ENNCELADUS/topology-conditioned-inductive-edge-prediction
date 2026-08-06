# Research Project: Topology-Conditioned Inductive Edge Prediction

## Overview

This repository supports an ICLR 2027 research project on **topology-conditioned
inductive edge prediction**. The task is to predict whether an edge exists between
two previously unseen nodes using frozen node features, without accessing the target
graph at inference time.

The project treats edge prediction as binary classification whose decisions should be
conditioned on locally generated topology. For a queried pair, the method constructs a
feature-derived local graph context and uses it to predict the edge probability. The
generated topology is intermediate context; the final output remains a binary edge
prediction for the queried pair.

## Motivation

Independent pairwise scorers can achieve reasonable edge-level metrics while their
scores assemble into graphs with implausible degree, clustering, or spectral structure.
This pair-to-topology gap motivates evaluating both edge-level performance and the
topology of the assembled predicted graph.

## Research Questions

1. Can local, feature-derived topology improve inductive edge prediction over
   independent pairwise scoring?
2. Can this improve assembled-graph topology without sacrificing edge-level quality?
3. Which parts of the method--candidate retrieval, topology construction, and
   topology-conditioned classification--are responsible for any improvement?

## Method and Evaluation

The active method line, EgoStitch, combines feature-based candidate retrieval, local
topology construction, and a topology-conditioned pair classifier. Experiments use
node-disjoint inductive splits and compare it with topology-blind and structure-aware
baselines under identical features and query sets.

Every result reports edge-level metrics (such as AUROC and AUPRC) together with
assembled-graph metrics, including density calibration and distances for degree,
clustering, and spectral structure. This prevents a method from being judged solely by
individual pair scores when its collective predictions form an unrealistic graph.

## Current Status

The benchmark pipeline, baseline models, and EgoStitch end-to-end implementation are
available in this repository. Historical fixed-seed G5 screens were engineering-valid
but did not establish a topology-conditioning gain; the current rev-3.2 line remains
an active research build. Scientific claims require the registered multi-seed
evaluation, with edge and graph metrics reported together.

## Further Reading

- `README.md` for project orientation and the current implementation status.
- `docs/01-blueprint.md` for the research questions and evaluation plan.
- `docs/04-model-proposal.md` and `docs/05-egostitch-spec.md` for method design and
  implementation constraints.
