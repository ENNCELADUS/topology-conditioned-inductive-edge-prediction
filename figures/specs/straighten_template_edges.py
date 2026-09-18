"""Post-step for figures/method_overview_v4: straight template edges, then the SVG.

The draw.io skill CLI forces every connector to orthogonal routing.  The motif-template
drawing inside the generator box needs plain straight lines, so this script rewrites the
edges whose stroke colour is the closure colour (#2E6FBF) or the bridge colour (#2A9D5C)
in the rendered .drawio: orthogonal routing and arrowheads removed, synthesised waypoints
dropped.  It then renders the SVG with the skill's own renderer (the CLI's drawio import
would re-normalise the styles).

Usage (from the repository root, after the CLI has written the .drawio):
    python3 figures/specs/straighten_template_edges.py
"""

from __future__ import annotations

import pathlib
import subprocess
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[2]
DRAWIO = ROOT / "figures" / "method_overview_v4.drawio"
SVG = ROOT / "figures" / "method_overview_v4.svg"
RENDERER = pathlib.Path.home() / ".claude" / "skills" / "drawio" / "scripts" / "svg" / "drawio-to-svg.js"
STRAIGHT_COLOURS = ("strokeColor=#2E6FBF", "strokeColor=#2A9D5C")
DROP_KEYS = ("edgeStyle", "rounded", "orthogonalLoop", "jettySize", "endArrow", "endFill", "endSize", "startArrow", "startFill", "startSize")


def straighten(style: str) -> str:
    """Return the style string without orthogonal routing or arrowheads."""
    kept = [tok for tok in style.split(";") if tok and tok.split("=")[0] not in DROP_KEYS]
    kept.extend(["endArrow=none", "startArrow=none"])
    return ";".join(kept) + ";"


def main() -> None:
    """Patch the .drawio in place and write the SVG."""
    tree = ET.parse(DRAWIO)
    changed = 0
    for cell in tree.iter("mxCell"):
        style = cell.get("style") or ""
        if cell.get("edge") != "1" or not any(c in style for c in STRAIGHT_COLOURS):
            continue
        cell.set("style", straighten(style))
        geometry = cell.find("mxGeometry")
        if geometry is not None:
            for arr in list(geometry.findall("Array")):
                geometry.remove(arr)
        changed += 1
    tree.write(DRAWIO, encoding="unicode", xml_declaration=True)
    script = (
        f"import {{drawioToSvg}} from '{RENDERER.as_uri()}';"
        "import {readFileSync, writeFileSync} from 'fs';"
        f"writeFileSync({str(SVG)!r}, drawioToSvg(readFileSync({str(DRAWIO)!r}, 'utf-8')));"
    )
    subprocess.run(["node", "--input-type=module", "-e", script], check=True)
    print(f"straightened {changed} template edges; wrote {SVG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
