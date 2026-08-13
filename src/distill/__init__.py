"""B1 training-time topology distillation: KD sampling, artifacts, dumper CLI, loss functions.

See ``docs/`` and the B1 implementation plan for the surrounding design
(Full-Ego Pooled Oracle teacher -> `NodeFactorBottleneck` student). This
package owns only the teacher-side artifact and the loss math; the student
architecture and trainer wiring live in ``src/model/egostitch/`` and
``src/train_egostitch.py``.
"""

from __future__ import annotations
