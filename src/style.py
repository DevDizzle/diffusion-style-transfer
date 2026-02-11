"""
Style Transfer Utilities
========================
CLIP-based style similarity scoring and IP-Adapter helpers for
applying artistic style to Stable Diffusion generations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class StyleScore:
    """Result of comparing a generated image to a style reference."""
    similarity: float        # Cosine similarity [-1, 1]
    style_name: str
    reference_path: str


class StyleEncoder:
    """
    Encodes images into CLIP embedding space for style comparison.
    Uses OpenCLIP ViT-H-14 for high-quality embeddings.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._preprocess = None

    def load(self) -> None:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        logger.info("StyleEncoder: loaded CLIP ViT-H-14")

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Return L2-normalized CLIP embedding for an image."""
        if self._model is None:
            self.load()
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        emb = self._model.encode_image(tensor)
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb

    @torch.inference_mode()
    def encode_text(self, text: str) -> torch.Tensor:
        """Return L2-normalized CLIP embedding for a text prompt."""
        if self._model is None:
            self.load()
        import open_clip

        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        tokens = tokenizer([text]).to(self.device)
        emb = self._model.encode_text(tokens)
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb


class StyleSimilarityScorer:
    """
    Compares generated images against one or more style references.
    Useful for ensuring brand-consistent artistic direction.
    """

    def __init__(self, device: str = "cuda"):
        self.encoder = StyleEncoder(device=device)
        self._style_cache: dict[str, torch.Tensor] = {}

    def load(self) -> None:
        self.encoder.load()

    def register_style(
        self, name: str, image_paths: list[str | Path]
    ) -> None:
        """
        Register a named style from one or more reference images.
        The style embedding is the mean of all reference embeddings.
        """
        embeddings = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            emb = self.encoder.encode_image(img)
            embeddings.append(emb)
        mean_emb = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
        mean_emb /= mean_emb.norm(dim=-1, keepdim=True)
        self._style_cache[name] = mean_emb
        logger.info("Registered style '%s' from %d images", name, len(embeddings))

    def register_style_from_text(self, name: str, description: str) -> None:
        """
        Register a style from a text description (e.g., "watercolor painting
        in the style of Studio Ghibli with soft pastel colors").
        """
        emb = self.encoder.encode_text(description)
        self._style_cache[name] = emb
        logger.info("Registered text-based style '%s'", name)

    @torch.inference_mode()
    def score(self, image: Image.Image, style_name: str) -> StyleScore:
        """Compute cosine similarity between image and named style."""
        if style_name not in self._style_cache:
            raise ValueError(f"Style '{style_name}' not registered")
        img_emb = self.encoder.encode_image(image)
        style_emb = self._style_cache[style_name]
        sim = (img_emb @ style_emb.T).item()
        return StyleScore(
            similarity=float(sim),
            style_name=style_name,
            reference_path="(cached)",
        )

    def score_all(self, image: Image.Image) -> list[StyleScore]:
        """Score image against all registered styles."""
        return [self.score(image, name) for name in self._style_cache]

    def best_match(self, image: Image.Image) -> Optional[StyleScore]:
        """Return the highest-scoring style for the image."""
        scores = self.score_all(image)
        return max(scores, key=lambda s: s.similarity) if scores else None


# ── Style-Conditioned Generation Helpers ─────────────────────────────────────

def load_ip_adapter(
    pipe,
    ip_adapter_model: str = "h94/IP-Adapter",
    weight_name: str = "ip-adapter-plus_sdxl_vit-h.safetensors",
    scale: float = 0.6,
):
    """
    Attach IP-Adapter to an existing StableDiffusionXL pipeline.

    Args:
        pipe: A loaded StableDiffusionXLPipeline instance.
        ip_adapter_model: HuggingFace repo for IP-Adapter weights.
        weight_name: Specific weight file within the repo.
        scale: Strength of style conditioning (0 = ignore, 1 = full).

    Returns:
        The modified pipeline with IP-Adapter loaded.
    """
    pipe.load_ip_adapter(
        ip_adapter_model,
        subfolder="sdxl_models",
        weight_name=weight_name,
    )
    pipe.set_ip_adapter_scale(scale)
    logger.info(
        "IP-Adapter loaded: %s (scale=%.2f)", weight_name, scale
    )
    return pipe


def prepare_style_image(
    image_path: str | Path,
    target_size: tuple[int, int] = (1024, 1024),
) -> Image.Image:
    """Load and resize a style reference image."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return img


def generate_with_style(
    pipe,
    prompt: str,
    style_image: Image.Image,
    negative_prompt: str = "",
    num_inference_steps: int = 40,
    guidance_scale: float = 7.5,
    ip_adapter_scale: float = 0.6,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Generate an image conditioned on both a text prompt and a style
    reference image via IP-Adapter.

    Args:
        pipe: SDXL pipeline with IP-Adapter loaded.
        prompt: Text description of the desired image.
        style_image: Reference image for style conditioning.
        negative_prompt: What to avoid in generation.
        num_inference_steps: Denoising steps.
        guidance_scale: Classifier-free guidance strength.
        ip_adapter_scale: IP-Adapter influence.
        seed: Random seed for reproducibility.

    Returns:
        Generated PIL Image.
    """
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=style_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]
