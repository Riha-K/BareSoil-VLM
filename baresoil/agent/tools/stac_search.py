"""STAC search tools for Sentinel-2 imagery."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def search_sentinel2_stac(
    bbox: Tuple[float, float, float, float],
    datetime_range: str,
    max_items: int = 5,
    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
) -> List[Dict[str, Any]]:
    """
    Search Planetary Computer STAC for Sentinel-2 L2A items.
    bbox: (min_lon, min_lat, max_lon, max_lat)
    datetime_range: e.g. '2024-01-01/2024-03-31'
    """
    try:
        from pystac_client import Client
    except ImportError:
        return [{
            "error": "pystac-client not installed",
            "hint": "pip install pystac-client planetary-computer",
            "bbox": bbox,
            "datetime": datetime_range,
        }]

    catalog = Client.open(stac_url)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox),
        datetime=datetime_range,
        max_items=max_items,
        query={"eo:cloud_cover": {"lt": 30}},
    )
    items = []
    for item in search.items():
        items.append({
            "id": item.id,
            "datetime": item.datetime.isoformat() if item.datetime else None,
            "bbox": item.bbox,
            "assets": list(item.assets.keys()),
            "self_href": item.self_href,
        })
    return items
