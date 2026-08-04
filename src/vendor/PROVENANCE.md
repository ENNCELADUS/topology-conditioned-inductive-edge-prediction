# Vendored-upstream provenance

| field | value |
| --- | --- |
| upstream URL | <https://github.com/LiamMa/GRIT> |
| pinned commit | `6c988ea600a606fbb49a2246c64a2d37396b3ab5` |
| upstream commit date | 2025-08-20T22:12:45+08:00 |
| vendored on | 2026-08-04 |
| license | MIT (copied to `src/vendor/grit_official/LICENSE`) |
| paper | Ma et al., *Graph Inductive Biases in Transformers without Message Passing* (ICML 2023), arXiv:2305.17589 |

## Reproducing the upstream tree to audit a shim diff

No copy of upstream is kept in this repo: nothing under it was ever imported at
runtime, so a 920 KB checkout bought nothing that the pinned commit plus the
sha256 table below does not. To check a recorded shim diff against the exact
bytes it was derived from:

```bash
git clone https://github.com/LiamMa/GRIT /tmp/GRIT
git -C /tmp/GRIT checkout 6c988ea600a606fbb49a2246c64a2d37396b3ab5
shasum -a 256 /tmp/GRIT/grit/layer/grit_layer.py   # must match the table below
diff <(tr -d '\r' < /tmp/GRIT/grit/layer/grit_layer.py) \
     src/vendor/grit_official/grit_layer.py        # upstream ships CRLF; ours is LF
```

The diff must show exactly the shim block recorded in that file's header and
nothing else. If the sha256 does not match, upstream moved the tag or force-pushed
and the vendored copy's provenance claim is void -- do not "fix" it silently.

## Files actually re-used

| upstream path | sha256 (upstream bytes) | vendored copy |
| --- | --- | --- |
| `grit/layer/grit_layer.py` | `52730905cd2e2c789bf2b113e257a6a3c13df548f739478c67dad9b331701ce5` | `src/vendor/grit_official/grit_layer.py` |

Read-only references (consulted, not copied):

- `grit/transform/rrwp.py` — RRWP definition: `pe_list = [I, P, P^2, ..., P^(k-1)]`
  with `P = D^-1 A` row-normalized; node-level absolute PE is `pe.diagonal()`, i.e.
  the per-node diagonal of each power. `src/model/egostitch/encoder/grit_gmt.py`
  mirrors this exactly, densely.
- `grit/encoder/rrwp_encoder.py` — `RRWPLinearEdgeEncoder.forward` with
  `pad_to_full_graph=True` (the setting every `configs/GRIT/*.yaml` uses) pads the
  relative-PE edge set to the **fully connected graph including self-loops**
  (`add_self_loops(..., fill_value=0.)` then `full_edge_index(...)`). That is why
  `grit_gmt.py` builds the complete `n_i^2` edge index per graph, diagonal included.

## Second vendored upstream

`src/vendor/set_transformer_official/` is byte-derived from a *different* upstream;
audit it by the same re-clone procedure, substituting this URL and commit:

| field | value |
| --- | --- |
| upstream URL | <https://github.com/juho-lee/set_transformer> |
| pinned commit | `73432c640ac78140496d6738416c54d32c686d65` |
| upstream commit date | 2020-02-11T12:12:08+09:00 |
| vendored on | 2026-08-04 |
| license | MIT (copied to `src/vendor/set_transformer_official/LICENSE`) |
| paper | Lee et al., *Set Transformer* (ICML 2019), arXiv:1810.00825 |

Only `modules.py`'s `MAB` and `PMA` are re-used; the shim diff against those exact
upstream bytes is recorded in that file's header.
