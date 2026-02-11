"""
Content Safety Module
=====================
NSFW detection, brand-consistency scoring via CLIP, and content rating.

Designed for production pipelines where every generated image must pass
safety and brand-alignment checks before delivery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ── Content Rating ──────────────────────────────────────────────────────────

class ContentRating(str, Enum):
    G = "G"
    PG = "PG"
    PG13 = "PG-13"
    RESTRICTED = "RESTRICTED"
    BLOCKED = "BLOCKED"


@dataclass
class SafetyResult:
    """Result of a full safety evaluation on a single image."""
    is_safe: bool
    nsfw_score: float
    content_rating: ContentRating
    brand_similarity: Optional[float] = None
    flagged_concepts: list[str] = field(default_factory=list)
    details: str = ""


# ── NSFW Classifier ─────────────────────────────────────────────────────────

class NSFWClassifier:
    """
    Wraps the Stable Diffusion safety_checker from HuggingFace Diffusers.
    Falls back to a CLIP-based zero-shot classifier if safety_checker
    is unavailable.
    """

    def __init__(self, device: str = "cuda", threshold: float = 0.85):
        self.device = device
        self.threshold = threshold
        self._checker = None
        self._feature_extractor = None
        self._clip_model = None
        self._clip_processor = None

    def load(self) -> None:
        """Load the diffusers safety checker."""
        try:
            from diffusers.pipelines.stable_diffusion import (
                StableDiffusionSafetyChecker,
            )
            from transformers import CLIPFeatureExtractor

            self._feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self._checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
                torch_dtype=torch.float16,
            ).to(self.device)
            logger.info("Loaded diffusers StableDiffusionSafetyChecker")
        except Exception as exc:
            logger.warning("Safety checker unavailable (%s); using CLIP fallback", exc)
            self._load_clip_fallback()

    def _load_clip_fallback(self) -> None:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        self._clip_model = model.to(self.device).eval()
        self._clip_processor = preprocess
        self._clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    @torch.inference_mode()
    def score(self, image: Image.Image) -> float:
        """Return NSFW probability (0 = safe, 1 = unsafe)."""
        if self._checker is not None:
            inputs = self._feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            np_image = np.array(image)[np.newaxis, ...]
            _, has_nsfw = self._checker(
                images=np_image,
                clip_input=inputs["pixel_values"],
            )
            return 1.0 if has_nsfw[0] else 0.0

        # CLIP zero-shot fallback
        if self._clip_model is None:
            self._load_clip_fallback()
        img_tensor = self._clip_processor(image).unsqueeze(0).to(self.device)
        safe_text = self._clip_tokenizer(["a safe family-friendly image"]).to(self.device)
        unsafe_text = self._clip_tokenizer(["an nsfw explicit adult image"]).to(self.device)

        img_feat = self._clip_model.encode_image(img_tensor)
        safe_feat = self._clip_model.encode_text(safe_text)
        unsafe_feat = self._clip_model.encode_text(unsafe_text)

        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        safe_feat /= safe_feat.norm(dim=-1, keepdim=True)
        unsafe_feat /= unsafe_feat.norm(dim=-1, keepdim=True)

        safe_sim = (img_feat @ safe_feat.T).item()
        unsafe_sim = (img_feat @ unsafe_feat.T).item()
        # Softmax over two classes
        nsfw_prob = np.exp(unsafe_sim) / (np.exp(safe_sim) + np.exp(unsafe_sim))
        return float(nsfw_prob)


# ── Brand Consistency Scorer ─────────────────────────────────────────────────

class BrandConsistencyScorer:
    """
    Measures how well a generated image aligns with a set of reference
    brand images using CLIP embedding cosine similarity.

    Use case: ensure generated content matches Disney's visual language,
    color palette, and thematic tone.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._preprocess = None
        self._reference_embeddings: Optional[torch.Tensor] = None

    def load(self) -> None:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        logger.info("Loaded CLIP ViT-H-14 for brand consistency scoring")

    @torch.inference_mode()
    def set_reference_images(self, image_paths: list[str | Path]) -> None:
        """Compute and cache embeddings for brand reference images."""
        if self._model is None:
            self.load()
        embeddings = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            tensor = self._preprocess(img).unsqueeze(0).to(self.device)
            emb = self._model.encode_image(tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb)
        self._reference_embeddings = torch.cat(embeddings, dim=0)
        logger.info("Cached %d brand reference embeddings", len(embeddings))

    @torch.inference_mode()
    def score(self, image: Image.Image) -> float:
        """
        Return mean cosine similarity to brand reference images.
        Range: [-1, 1], higher = more brand-consistent.
        """
        if self._reference_embeddings is None:
            logger.warning("No reference images set; returning 0.0")
            return 0.0
        if self._model is None:
            self.load()
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        emb = self._model.encode_image(tensor)
        emb /= emb.norm(dim=-1, keepdim=True)
        similarities = (emb @ self._reference_embeddings.T).squeeze(0)
        return float(similarities.mean().item())


# ── Content Rating System ────────────────────────────────────────────────────

class ContentRatingSystem:
    """
    Assigns a content rating (G / PG / PG-13 / RESTRICTED / BLOCKED)
    based on CLIP zero-shot classification against concept categories.
    """

    CONCEPT_BUCKETS: dict[str, list[str]] = {
        "violence": [
            "a violent image", "fighting", "weapons", "blood",
        ],
        "adult": [
            "nudity", "sexually suggestive content", "explicit content",
        ],
        "scary": [
            "a scary horror image", "dark disturbing imagery",
        ],
        "wholesome": [
            "a happy family-friendly image", "cute cartoon characters",
            "a bright colorful wholesome scene",
        ],
    }

    def __init__(
        self,
        device: str = "cuda",
        rating_thresholds: Optional[dict[str, float]] = None,
        blocked_concepts: Optional[list[str]] = None,
    ):
        self.device = device
        self.thresholds = rating_thresholds or {"G": 0.95, "PG": 0.80, "PG13": 0.60}
        self.blocked_concepts = set(blocked_concepts or [])
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def load(self) -> None:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-H-14")

    @torch.inference_mode()
    def rate(self, image: Image.Image) -> tuple[ContentRating, dict[str, float], list[str]]:
        """
        Returns (rating, concept_scores, flagged_concepts).
        """
        if self._model is None:
            self.load()

        img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        img_feat = self._model.encode_image(img_tensor)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)

        concept_scores: dict[str, float] = {}
        for concept, prompts in self.CONCEPT_BUCKETS.items():
            tokens = self._tokenizer(prompts).to(self.device)
            text_feat = self._model.encode_text(tokens)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ text_feat.T).squeeze(0)
            concept_scores[concept] = float(sims.max().item())

        # Flag blocked concepts
        flagged = [c for c in self.blocked_concepts if concept_scores.get(c, 0) > 0.25]

        if flagged:
            return ContentRating.BLOCKED, concept_scores, flagged

        # Wholesome score determines rating
        wholesome = concept_scores.get("wholesome", 0.0)
        negative_max = max(
            concept_scores.get("violence", 0),
            concept_scores.get("adult", 0),
            concept_scores.get("scary", 0),
        )

        safety_score = wholesome - negative_max  # crude but effective

        if safety_score >= self.thresholds["G"]:
            rating = ContentRating.G
        elif safety_score >= self.thresholds["PG"]:
            rating = ContentRating.PG
        elif safety_score >= self.thresholds["PG13"]:
            rating = ContentRating.PG13
        else:
            rating = ContentRating.RESTRICTED

        return rating, concept_scores, flagged


# ── Unified Safety Evaluator ────────────────────────────────────────────────

class SafetyEvaluator:
    """
    Orchestrates all safety checks for a single image.
    """

    def __init__(
        self,
        device: str = "cuda",
        nsfw_threshold: float = 0.85,
        brand_threshold: float = 0.30,
        blocked_concepts: Optional[list[str]] = None,
        rating_thresholds: Optional[dict[str, float]] = None,
    ):
        self.nsfw = NSFWClassifier(device=device, threshold=nsfw_threshold)
        self.brand = BrandConsistencyScorer(device=device)
        self.rating_system = ContentRatingSystem(
            device=device,
            rating_thresholds=rating_thresholds,
            blocked_concepts=blocked_concepts,
        )
        self.nsfw_threshold = nsfw_threshold
        self.brand_threshold = brand_threshold

    def load_all(self) -> None:
        self.nsfw.load()
        self.brand.load()
        self.rating_system.load()

    def evaluate(
        self,
        image: Image.Image,
        check_brand: bool = True,
    ) -> SafetyResult:
        nsfw_score = self.nsfw.score(image)
        rating, concept_scores, flagged = self.rating_system.rate(image)
        brand_sim = self.brand.score(image) if check_brand else None

        is_safe = (
            nsfw_score < self.nsfw_threshold
            and rating not in (ContentRating.BLOCKED, ContentRating.RESTRICTED)
            and (brand_sim is None or brand_sim >= self.brand_threshold)
        )

        details_parts = [f"NSFW={nsfw_score:.3f}"]
        if brand_sim is not None:
            details_parts.append(f"brand_sim={brand_sim:.3f}")
        details_parts.append(f"rating={rating.value}")
        if flagged:
            details_parts.append(f"flagged={flagged}")

        return SafetyResult(
            is_safe=is_safe,
            nsfw_score=nsfw_score,
            content_rating=rating,
            brand_similarity=brand_sim,
            flagged_concepts=flagged,
            details=" | ".join(details_parts),
        )
