"""VLM inference wrapper for BareSoilDial / EarthDial."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class BareSoilVLM:
    """Thin wrapper around InternVLChatModel for bare-soil queries."""

    def __init__(
        self,
        checkpoint: str = "./checkpoints/BareSoilDial_4B_RGB_v01",
        device: str = "cuda",
        load_in_8bit: bool = False,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.load_in_8bit = load_in_8bit
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer
        from earthdial.model.internvl_chat import InternVLChatModel

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, trust_remote_code=True, use_fast=False,
        )
        kwargs = {}
        if self.load_in_8bit:
            kwargs["load_in_8bit"] = True
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        self._model = InternVLChatModel.from_pretrained(
            self.checkpoint, low_cpu_mem_usage=True, **kwargs,
        ).eval()
        if not self.load_in_8bit:
            self._model = self._model.to(self.device)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def chat(
        self,
        image_path: Union[str, Path],
        question: str,
        max_new_tokens: int = 128,
    ) -> str:
        if not self.ready:
            self.load()
        import torch
        from PIL import Image
        from earthdial.train.dataset import build_transform, dynamic_preprocess

        image = Image.open(image_path).convert("RGB")
        transform = build_transform(is_train=False, input_size=224)
        images = dynamic_preprocess(image, image_size=224, use_thumbnail=True, max_num=6)
        pixel_values = torch.stack([transform(img) for img in images])
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)

        if not question.strip().startswith("[baresoil]"):
            question = f"[baresoil] [hr_rgb_0.5] {question}"
        if "<image>" not in question:
            question = f"{question}\n<image>"

        gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
        return self._model.chat(
            tokenizer=self._tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=gen_cfg,
            verbose=False,
        )
