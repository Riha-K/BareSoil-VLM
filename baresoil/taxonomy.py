"""
Unified bare-soil / barren-land taxonomy mapping across RS label schemes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

# Canonical unified classes for BareSoil-Bench and BareSoil-Instruct
UNIFIED_CLASSES = [
    "bare_soil",
    "sparse_vegetation",
    "desert_sand",
    "bare_rock_paved",
    "burnt_barren",
    "agricultural_fallow",
    "non_bare",
]

BARE_POSITIVE_CLASSES: Set[str] = {
    "bare_soil",
    "sparse_vegetation",
    "desert_sand",
    "bare_rock_paved",
    "burnt_barren",
    "agricultural_fallow",
}

# AID scene labels -> unified
AID_TO_UNIFIED: Dict[str, str] = {
    "BareLand": "bare_soil",
    "Desert": "desert_sand",
    "Meadow": "sparse_vegetation",
    "Beach": "desert_sand",
    "Mountain": "non_bare",
    "Farmland": "agricultural_fallow",
}

# LCZ42 labels -> unified
LCZ_TO_UNIFIED: Dict[str, str] = {
    "bare soil or sand": "bare_soil",
    "bare rock or paved": "bare_rock_paved",
    "low plants": "sparse_vegetation",
    "scrub": "sparse_vegetation",
    "bush": "sparse_vegetation",
    "sparsely built": "non_bare",
    "heavy industry": "non_bare",
    "water": "non_bare",
}

# CORINE 19-class (BigEarthNet) -> unified
CORINE19_TO_UNIFIED: Dict[str, str] = {
    "Beaches, dunes, sands": "desert_sand",
    "Natural grassland and sparsely vegetated areas": "sparse_vegetation",
    "Moors, heathland and sclerophyllous vegetation": "sparse_vegetation",
    "Arable land": "agricultural_fallow",
    "Pastures": "sparse_vegetation",
    "Industrial or commercial units": "non_bare",
    "Urban fabric": "non_bare",
}

# CORINE 43-class subset
CORINE43_TO_UNIFIED: Dict[str, str] = {
    "Beaches, dunes, sands": "desert_sand",
    "Bare rock": "bare_rock_paved",
    "Sparsely vegetated areas": "sparse_vegetation",
    "Burnt areas": "burnt_barren",
    "Natural grassland": "sparse_vegetation",
    "Moors and heathland": "sparse_vegetation",
    "Pastures": "sparse_vegetation",
}

# ESA WorldCover -> unified
WORLDCOVER_TO_UNIFIED: Dict[str, str] = {
    "Bare / sparse vegetation": "sparse_vegetation",
    "Herbaceous vegetation": "sparse_vegetation",
    "Cropland": "agricultural_fallow",
    "Built-up": "non_bare",
    "Water": "non_bare",
}

# Dynamic World -> unified
DYNAMICWORLD_TO_UNIFIED: Dict[str, str] = {
    "Bare ground": "bare_soil",
    "Scrub/shrub": "sparse_vegetation",
    "Built": "non_bare",
    "Water": "non_bare",
}


@dataclass(frozen=True)
class TaxonomyMapping:
    source: str
    raw_label: str
    unified: str
    is_bare_positive: bool


def map_label(raw: str, scheme: str) -> Optional[str]:
    """Map a source label to unified taxonomy."""
    raw_norm = raw.strip()
    tables = {
        "aid": AID_TO_UNIFIED,
        "lcz": LCZ_TO_UNIFIED,
        "corine19": CORINE19_TO_UNIFIED,
        "corine43": CORINE43_TO_UNIFIED,
        "worldcover": WORLDCOVER_TO_UNIFIED,
        "dynamicworld": DYNAMICWORLD_TO_UNIFIED,
    }
    table = tables.get(scheme.lower())
    if table is None:
        return None
    # case-insensitive lookup
    for k, v in table.items():
        if k.lower() == raw_norm.lower():
            return v
    return table.get(raw_norm)


def is_bare_positive(unified: Optional[str]) -> bool:
    return unified in BARE_POSITIVE_CLASSES


def unified_display_name(unified: str) -> str:
    return unified.replace("_", " ")


def all_scheme_mappings() -> List[TaxonomyMapping]:
    rows: List[TaxonomyMapping] = []
    for scheme, table in [
        ("AID", AID_TO_UNIFIED),
        ("LCZ42", LCZ_TO_UNIFIED),
        ("CORINE19", CORINE19_TO_UNIFIED),
        ("CORINE43", CORINE43_TO_UNIFIED),
        ("WorldCover", WORLDCOVER_TO_UNIFIED),
        ("DynamicWorld", DYNAMICWORLD_TO_UNIFIED),
    ]:
        for raw, unified in table.items():
            rows.append(
                TaxonomyMapping(
                    source=scheme,
                    raw_label=raw,
                    unified=unified,
                    is_bare_positive=unified in BARE_POSITIVE_CLASSES,
                )
            )
    return rows
