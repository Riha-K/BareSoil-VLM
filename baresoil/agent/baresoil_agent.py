"""
BareSoil-Agent: agentic monitoring pipeline for bare/barren land.
Plan -> STAC search -> spectral indices -> BareSoilDial VLM -> report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from baresoil.agent.tools.report import generate_markdown_report
from baresoil.agent.tools.spectral import bsi_sentinel2, ndvi, summarize_bare_soil_indices
from baresoil.agent.tools.stac_search import search_sentinel2_stac
from baresoil.agent.tools.vlm_inference import BareSoilVLM


@dataclass
class AgentConfig:
    checkpoint: str = "./checkpoints/BareSoilDial_4B_RGB_v01"
    output_dir: Path = field(default_factory=lambda: Path("baresoil/data/agent_outputs"))
    vlm_enabled: bool = True


class BareSoilAgent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._vlm: Optional[BareSoilVLM] = None

    def plan(self, user_query: str) -> List[str]:
        """Decompose user query into tool steps."""
        steps = ["stac_search", "spectral_analysis"]
        if self.config.vlm_enabled:
            steps.append("vlm_interpret")
        steps.append("generate_report")
        return steps

    def run(
        self,
        user_query: str,
        aoi_name: str,
        bbox: Tuple[float, float, float, float],
        datetime_range: str = "2024-01-01/2024-03-31",
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        steps = self.plan(user_query)
        result: Dict[str, Any] = {"query": user_query, "steps": steps, "aoi": aoi_name}

        # 1. STAC
        stac_items = search_sentinel2_stac(bbox=bbox, datetime_range=datetime_range)
        result["stac_items"] = stac_items

        # 2. Spectral (synthetic if no raster loaded)
        spectral_summary = self._spectral_from_image_or_demo(image_path)
        result["spectral"] = spectral_summary

        # 3. VLM
        vlm_responses = []
        if self.config.vlm_enabled and image_path and Path(image_path).exists():
            if self._vlm is None:
                self._vlm = BareSoilVLM(checkpoint=self.config.checkpoint)
            for q in [
                "[classify] Is bare soil dominant? Options: bare soil, sparse vegetation, non bare.",
                "Describe barren or bare areas and possible confusers (roads, rooftops).",
            ]:
                try:
                    ans = self._vlm.chat(image_path, q)
                    vlm_responses.append({"question": q, "answer": ans})
                except Exception as e:
                    vlm_responses.append({"question": q, "answer": f"[VLM unavailable: {e}]"})
        else:
            vlm_responses.append({
                "question": user_query,
                "answer": (
                    f"Spectral bare fraction: {spectral_summary.get('bare_fraction', 0):.1%}. "
                    f"Found {len(stac_items)} Sentinel-2 scenes for {datetime_range}."
                ),
            })
        result["vlm"] = vlm_responses

        # 4. Report
        report_path = self.config.output_dir / f"{aoi_name.replace(' ', '_')}_report.md"
        md = generate_markdown_report(
            query=user_query,
            aoi_name=aoi_name,
            stac_items=stac_items,
            spectral_summary=spectral_summary,
            vlm_responses=vlm_responses,
            output_path=report_path,
        )
        result["report_path"] = str(report_path)
        result["report_preview"] = md[:500]
        return result

    def _spectral_from_image_or_demo(self, image_path: Optional[str]) -> Dict[str, Any]:
        try:
            import numpy as np
            if image_path and Path(image_path).exists():
                # Demo: treat grayscale as pseudo-NDVI for pipeline test
                from PIL import Image
                arr = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
                ndvi_map = arr * 0.5
                bsi_map = 1.0 - arr * 0.6
                return summarize_bare_soil_indices(ndvi_map, bsi_map)
        except Exception:
            pass
        return {
            "bare_fraction": 0.34,
            "mean_ndvi": 0.18,
            "mean_bsi": 0.42,
            "note": "demo values — load Sentinel-2 bands for real analysis",
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BareSoil-Agent demo")
    parser.add_argument("--query", default="Find bare soil areas and explain spectral evidence")
    parser.add_argument("--aoi", default="Demo_AOI")
    parser.add_argument("--bbox", default="73.7,18.4,74.0,18.7", help="lon_min,lat_min,lon_max,lat_max")
    parser.add_argument("--datetime", default="2024-01-01/2024-03-31")
    parser.add_argument("--image", default=None)
    parser.add_argument("--no-vlm", action="store_true")
    args = parser.parse_args()

    bbox = tuple(float(x) for x in args.bbox.split(","))
    agent = BareSoilAgent(AgentConfig(vlm_enabled=not args.no_vlm))
    out = agent.run(args.query, args.aoi, bbox, args.datetime, args.image)
    print(json.dumps({k: v for k, v in out.items() if k != "report_preview"}, indent=2))
    print(f"\nReport: {out['report_path']}")


if __name__ == "__main__":
    main()
