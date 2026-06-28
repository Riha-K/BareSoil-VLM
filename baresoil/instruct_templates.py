"""
Instruction templates for BareSoil-Instruct dataset generation.
"""

from __future__ import annotations

from typing import List

UNIFIED_OPTIONS = [
    "bare soil",
    "sparse vegetation",
    "desert sand",
    "bare rock or paved",
    "burnt barren",
    "agricultural fallow",
    "non bare",
]

CLASSIFY_TEMPLATES_RGB = [
    "[baresoil] [hr_rgb_0.5] [classify] <image>\nClassify land cover. Options: {options}.\nAnswer in one word or short phrase.",
    "[hr_rgb_0.5] [classify] <image>\nWhat is the dominant surface type? Options: {options}.",
    "[baresoil] [classify] <image>\nIs this region bare or vegetated? Choose from: {options}.",
    "[hr_rgb_0.5] <image>\n[baresoil] Identify the barren land category. Classes: {options}.",
]

CLASSIFY_TEMPLATES_MS = [
    "[baresoil] [s2_ms_10] [classify] <image>\nUsing multispectral bands, classify: {options}.",
    "[s2_ms_10] [classify] <image>\nSpectral land cover (bare soil focus). Options: {options}.",
]

VQA_TEMPLATES = [
    "[baresoil] [hr_rgb_0.5] <image>\nIs bare soil or barren land visible in this image? Answer yes or no.",
    "[hr_rgb_0.5] <image>\n[baresoil] What percentage of the scene appears to be bare ground? (none / low / medium / high)",
    "[baresoil] [hr_rgb_0.5] <image>\nCould this scene be confused with urban built-up from RGB alone? Explain briefly.",
    "[s2_ms_10] <image>\n[baresoil] Which spectral index would best confirm bare soil here: NDVI, BSI, or NDBI?",
]

CAPTION_TEMPLATES = [
    "[baresoil] [caption] [hr_rgb_0.5] <image>\nDescribe bare or barren areas in this remote sensing image.",
    "[caption] [hr_rgb_0.5] <image>\nProvide a caption focusing on soil exposure and land cover.",
]

BINARY_VQA_YES = "yes"
BINARY_VQA_NO = "no"


def format_options(options: List[str] | None = None) -> str:
    opts = options or UNIFIED_OPTIONS
    return ", ".join(opts)


def expand_classify_templates(modality: str = "rgb") -> List[str]:
    templates = CLASSIFY_TEMPLATES_MS if modality == "ms" else CLASSIFY_TEMPLATES_RGB
    opts = format_options()
    return [t.format(options=opts) for t in templates]


def expand_all_templates() -> dict:
    return {
        "classify_rgb": expand_classify_templates("rgb"),
        "classify_ms": expand_classify_templates("ms"),
        "vqa": list(VQA_TEMPLATES),
        "caption": list(CAPTION_TEMPLATES),
    }
