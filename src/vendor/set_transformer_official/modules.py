"""Vendored `MAB` + `PMA` from the official Set Transformer reference code.

Upstream: https://github.com/juho-lee/set_transformer @
`73432c640ac78140496d6738416c54d32c686d65` (2020-02-11), file `modules.py`,
MIT licensed -- see `LICENSE` beside this file. Paper: Lee et al.,
*Set Transformer: A Framework for Attention-based Permutation-Invariant Neural
Networks* (ICML 2019), arXiv:1810.00825.

Only `MAB` and `PMA` are carried. `PMA` composes `MAB` alone -- it does **not**
use `SAB`/`ISAB`, so those are not vendored (verified against the upstream file:
`PMA.__init__` builds one `MAB` and nothing else).

**Do not restyle the class bodies.** As with `grit_official/grit_layer.py`, the
point is that a diff against upstream shows exactly the changes recorded here.
`# ruff: noqa` / `# mypy: ignore-errors` keep this repo's style rules off the
vendored bodies.

Recorded shim diff:

(0) `import torch.nn.functional as F` / `import math` / `import torch` /
    `import torch.nn as nn` are reordered only insofar as this docstring now
    precedes them. No import is added or removed.

(1) `SAB` and `ISAB` are omitted (unused; see above). `MAB`'s and `PMA`'s bodies
    are otherwise upstream's, character for character, apart from (2).

(2) **Key-padding mask support, added.** The official `MAB.forward(Q, K)` has no
    mask: upstream's benchmarks are fixed-cardinality point clouds. Our sets are
    padded node batches, so `MAB.forward` and `PMA.forward` gain an optional
    trailing `mask=None` argument, shape ``(B, n_keys)``, ``True`` = valid key.
    It is applied as ``-inf`` added to the pre-softmax attention logits, which
    is the standard, exactly-zero-attention formulation:

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)

    becomes a `masked_fill` on the logits before that softmax. Two details:

    - The heads are folded into the *batch* axis upstream
      (``torch.cat(Q.split(dim_split, 2), 0)``), so the mask is repeated
      ``num_heads`` times along dim 0 to line up. This is the one place a naive
      mask implementation goes silently wrong.
    - **All-invalid rows.** A row whose every key is masked would make the
      softmax ``exp(-inf - (-inf)) = nan``. Such rows are detected and their
      mask replaced by all-valid *for the softmax only*; with the caller
      zeroing padded key rows (as `GritGmtEncoder` does), the resulting output
      is the bias-only, finite value rather than `nan`. Returning `nan` for an
      empty set would poison an entire training step, so failing soft here --
      and loudly documenting it -- is the right trade.

    ``mask=None`` reproduces upstream's arithmetic exactly (no `masked_fill` is
    applied at all, not merely an all-True one), so the unmasked path is
    bit-identical to the reference implementation.
"""

# ruff: noqa
# mypy: ignore-errors

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # ---- shim (2): optional key-padding mask -----------------------------
        logits = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        if mask is not None:
            # `mask` is (B, n_keys); heads were folded into dim 0 above.
            keep = mask.to(dtype=torch.bool)
            keep = torch.where(
                keep.any(dim=1, keepdim=True), keep, torch.ones_like(keep)
            )
            keep = keep.repeat(self.num_heads, 1).unsqueeze(1)
            logits = logits.masked_fill(~keep, float("-inf"))
        A = torch.softmax(logits, 2)
        # ---- end shim (2) ----------------------------------------------------
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, mask)
