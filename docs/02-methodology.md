# Methodology: Design Open

**Status (2026-08-13):** no formal model design has been selected or specified.

This document records the constraints for choosing a method. It does not define an
architecture, topology representation, training objective, baseline ladder, or
implementation plan.

## 1. Fixed Method Boundary

The primary task is fully inductive and node-disjoint. It receives exactly the frozen
intrinsic endpoint features $(x_u,x_v)$ and returns a symmetric binary edge probability:

$$
p_{uv}=P(Y_{uv}=1\mid x_u,x_v)=p_{vu}.
$$

Any future method must satisfy these requirements:

1. The hidden test graph, observed test neighbors, degrees, and graph statistics are
   not inference inputs.
2. Training topology may provide supervision, but a strict endpoint-only method must
   infer any test-time context from $(x_u,x_v)$.
3. Inferred topology, if used, is intermediate context for deciding
   $\operatorname{edge}(u,v)$, not the prediction target.
4. Retrieval, grounding, prototypes, or external identities define a separately
   labeled support condition, must disclose that support, and are not strict
   endpoint-only methods.
5. Pair probabilities remain the model output. Cross-pair coupling, if proposed, must
   be specified and fixed without test topology; graph assembly remains an evaluation
   operation.

These are task and evidence constraints, not a model specification.

## 2. Open Design Questions

The following choices remain unresolved:

- whether topology conditioning is needed beyond a strong endpoint-only scorer;
- whether useful structural information should be represented explicitly, latently,
  deterministically, probabilistically, or only through a training objective;
- how any structural representation should be encoded and coupled to the edge decision;
- which losses, sampling procedure, and computational budget are justified;
- whether an endpoint-only method can improve edge and assembled-graph metrics together.

Literature categories and historical implementations are comparison evidence only.
They do not select a method or establish a formal design.

## 3. Requirements for a Formal Design

A future design can be called formal only when one document fixes:

1. the inference inputs and outputs;
2. the intermediate representation and every component that consumes it;
3. the training targets, losses, sampling, and queried-edge masking rules;
4. the data boundaries and leakage controls;
5. the evaluation-matched controls and ablations;
6. the expected compute path and reproducibility contract.

Until then, candidate code, diagrams, loss decompositions, and experiment ideas are
proposals or diagnostics rather than the project method.

## 4. Sources of Truth

- [01-project-definition.md](01-project-definition.md) defines the research problem,
  hypotheses, and related-work boundaries.
- [03-experiment-protocol.md](03-experiment-protocol.md) defines evidence classes,
  evaluation, reporting, and the current experimental record.
- Historical proposals and implemented comparison arms do not override either document.
