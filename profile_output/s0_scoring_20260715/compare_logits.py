"""Compare row-aligned logits from two S0 score artifacts."""

from __future__ import annotations

import argparse
import json

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("reference")
parser.add_argument("candidate")
args = parser.parse_args()

with np.load(args.reference, allow_pickle=False) as reference, np.load(
    args.candidate, allow_pickle=False
) as candidate:
    difference = np.abs(
        reference["logit"].astype(np.float64) - candidate["logit"].astype(np.float64)
    )
    summary = {
        "rows": int(difference.size),
        "pairs_equal": bool(
            np.array_equal(reference["u_idx"], candidate["u_idx"])
            and np.array_equal(reference["v_idx"], candidate["v_idx"])
        ),
        "max_abs_logit_diff": float(difference.max(initial=0.0)),
        "mean_abs_logit_diff": float(difference.mean()) if difference.size else 0.0,
        "p99_abs_logit_diff": (
            float(np.quantile(difference, 0.99)) if difference.size else 0.0
        ),
        "allclose_atol_0_01": bool(
            np.allclose(reference["logit"], candidate["logit"], rtol=0.0, atol=0.01)
        ),
    }

print(json.dumps(summary, sort_keys=True))
