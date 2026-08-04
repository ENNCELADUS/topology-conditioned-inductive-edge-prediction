"""Third-party code vendored under recorded, minimal shim diffs.

Every subpackage here is a near-byte-identical copy of an official upstream
implementation, kept out of `src/model/` so the boundary between "ours" and
"theirs" is a directory boundary rather than a comment. Each vendored module's
own docstring records its upstream URL, pinned commit, license, and the exact
line-by-line diff applied; each subpackage carries the upstream `LICENSE`.

`PROVENANCE.md` beside this file is the index; it also records how to re-clone
each upstream at its pinned commit to audit a shim diff.

Nothing in here may be reformatted or re-linted: the shim diffs are only
auditable while the surrounding bytes stay upstream's.
"""
