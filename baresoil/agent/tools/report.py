"""Report generation for BareSoil-Agent."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_markdown_report(
    query: str,
    aoi_name: str,
    stac_items: List[Dict],
    spectral_summary: Dict[str, Any],
    vlm_responses: List[Dict[str, str]],
    output_path: Optional[Path] = None,
) -> str:
    lines = [
        f"# BareSoil Monitoring Report",
        f"",
        f"**Generated:** {datetime.utcnow().isoformat()}Z",
        f"**AOI:** {aoi_name}",
        f"**Query:** {query}",
        f"",
        f"## STAC Search Results",
        f"",
    ]
    if stac_items and "error" not in stac_items[0]:
        for item in stac_items[:5]:
            lines.append(f"- `{item.get('id')}` — {item.get('datetime')}")
    else:
        lines.append(f"- _(STAC unavailable: {stac_items[0].get('error', 'no items')} )_")

    lines.extend([
        f"",
        f"## Spectral Analysis",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Bare fraction (NDVI+BSI) | {spectral_summary.get('bare_fraction', 0):.2%} |",
        f"| Mean NDVI | {spectral_summary.get('mean_ndvi', 0):.3f} |",
        f"| Mean BSI | {spectral_summary.get('mean_bsi', 0):.3f} |",
        f"",
        f"## VLM Interpretation",
        f"",
    ])
    for i, resp in enumerate(vlm_responses, 1):
        lines.append(f"### Q{i}: {resp.get('question', '')}")
        lines.append(f"{resp.get('answer', '')}")
        lines.append("")

    lines.extend([
        f"## Data Sources",
        f"- Sentinel-2 L2A via Microsoft Planetary Computer STAC",
        f"- BareSoilDial / EarthDial VLM",
        f"",
    ])

    md = "\n".join(lines)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md, encoding="utf-8")
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"name": aoi_name, "bare_fraction": spectral_summary.get("bare_fraction")},
                "geometry": None,
            }],
        }
        output_path.with_suffix(".geojson").write_text(json.dumps(geojson, indent=2))
    return md
