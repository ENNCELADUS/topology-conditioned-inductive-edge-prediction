"""B1 training-time knowledge distillation: teacher-target artifact, dumper CLI, loss functions.

See ``docs/`` and the B1 implementation plan for the surrounding design: a
Full-Ego Pooled Oracle teacher dumps one target row per official training
row (plus a V_val diagnostic block) via ``src.distill.teacher_targets``,
stored by ``src.distill.artifacts``. This package owns only the teacher-side
artifact and the same-batch KD loss math (``src.distill.losses``); the KD
arms and trainer wiring live in ``src/train_b0.py``.
"""

from __future__ import annotations
