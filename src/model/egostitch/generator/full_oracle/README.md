# Full-neighborhood oracle

For a query `q = (u, v)`, this diagnostic generator deletes the undirected
query edge from the bound truth graph, forms

`U_q = {u, v} union N_{G-q}(u) union N_{G-q}(v)`,

and emits the exact induced graph `(G - q)[U_q]`. Thus every endpoint neighbor
and every other ground-truth positive edge among those nodes is visible.

Each node has five channels: source endpoint, destination endpoint, source
neighbor, destination neighbor, and existence. A shared neighbor activates both
neighbor channels. The single adjacency relation is the binary truth-edge
relation.

Graphs have variable node count and are padded only to the largest graph in the
current batch. GRIT still constructs dense pair features/attention, so memory
and compute scale quadratically with the largest local graph in that batch.

This component reads held-out positive topology and is therefore
**diagnostic-only**. It measures the encoder/classifier ceiling under complete
local truth; it is not a protocol-clean inductive model.
