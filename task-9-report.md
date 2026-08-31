# Task 9 fix round 1

Added review-driven characterization regressions in `tests/test_b0_topo_gen.py`; production code was
already correct and was not changed. The tests passed on their first execution, so no RED result is
claimed.

- Pinned `_u` conditioning to the ordered 64-D `concat(Fourier(t), Fourier(t-r))` tail and rejected
  the symmetric duplicated-average alternative.
- Controlled the noise draw, both uniform draws, and JVP to pin the mixed `r=t`/`r=min` branches,
  interpolation, tangent, training `create_graph`, detached identity target, and adaptive scalar loss.
- Deep-compared the parsed iMF YAML with the EDM YAML after exactly the three approved edits.

Verification:

- Focused regressions: 3 passed, 18 deselected.
- Full `tests/test_b0_topo_gen.py`: 21 passed.
- Ruff: passed for `tests/test_b0_topo_gen.py`.
- Mypy: passed for `tests/test_b0_topo_gen.py`.
- `git diff --check`: passed.

No DDP, H20, or external reference lookup was performed.
