"""
End-to-End Generation Pipeline
===============================
prompt → safety check → generate → post-process → safety verify → output

Orchestrates Stable Diffusion XL with integrated content safety,
brand consistency scoring, and optional style transfer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml
from PIL import Image, ImageFilter

from .safety import ContentRating, SafetyEvaluator, SafetyResult
from .style import (
    StyleSimilarityScorer,
    generate_with_style,
    load_ip_adapter,
    prepare_style_image,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Encapsulates everything needed for a single generation."""
    prompt: str
    negative_prompt: str = ""
    seed: Optional[int] = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 40
    height: int = 1024
    width: int = 1024
    style_image_path: Optional[str] = None
    ip_adapter_scale: float = 0.6
    use_refiner: bool = True
    high_noise_frac: float = 0.8


@dataclass
class GenerationResult:
    """Full result including image, safety info, and metadata."""
    image: Image.Image
    safety: SafetyResult
    prompt: str
    seed: int
    generation_time_s: float
    style_scores: dict[str, float] = field(default_factory=dict)
    was_filtered: bool = False
    filter_reason: str = ""


class DiffusionPipeline:
    """
    Production-grade text-to-image pipeline with safety guardrails.

    Features:
    - SDXL base + refiner ensemble of expert denoisers
    - Pre-generation prompt safety screening
    - Post-generation NSFW & content rating checks
    - Brand consistency scoring via CLIP
    - Optional IP-Adapter style conditioning
    """

    def __init__(
        self,
        config_path: str = "configs/model_config.yaml",
        device: str = "cuda",
        brand_reference_images: Optional[list[str]] = None,
    ):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = device
        self.pipe = None
        self.refiner = None
        self._ip_adapter_loaded = False

        # Safety
        safety_cfg = self.config.get("safety", {})
        self.safety = SafetyEvaluator(
            device=device,
            nsfw_threshold=safety_cfg.get("nsfw_threshold", 0.85),
            brand_threshold=safety_cfg.get("brand_consistency_threshold", 0.30),
            blocked_concepts=safety_cfg.get("blocked_concepts", []),
            rating_thresholds=safety_cfg.get("content_rating_levels"),
        )

        # Style scoring
        self.style_scorer = StyleSimilarityScorer(device=device)
        self._brand_refs = brand_reference_images or []

    def load(self) -> None:
        """Load all models into GPU memory."""
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

        model_cfg = self.config["model"]
        logger.info("Loading SDXL base: %s", model_cfg["model_id"])

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_cfg["model_id"],
            torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "float16")),
            variant=model_cfg.get("variant", "fp16"),
            use_safetensors=model_cfg.get("use_safetensors", True),
        ).to(self.device)

        # Configure scheduler
        sched_cfg = self.config.get("scheduler", {})
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            solver_order=sched_cfg.get("solver_order", 2),
            use_karras_sigmas=sched_cfg.get("use_karras_sigmas", True),
        )

        # Optionally load refiner
        refiner_id = model_cfg.get("refiner_id")
        if refiner_id:
            from diffusers import StableDiffusionXLImg2ImgPipeline

            logger.info("Loading SDXL refiner: %s", refiner_id)
            self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_id,
                torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "float16")),
                variant=model_cfg.get("variant", "fp16"),
                use_safetensors=True,
            ).to(self.device)

        # Load safety models
        self.safety.load_all()

        # Set brand references if provided
        if self._brand_refs:
            self.safety.brand.set_reference_images(self._brand_refs)

        logger.info("Pipeline fully loaded and ready")

    def _screen_prompt(self, prompt: str) -> tuple[bool, str]:
        """
        Pre-generation prompt screening using blocked concept keywords.
        Returns (is_safe, reason).
        """
        blocked = self.config.get("safety", {}).get("blocked_concepts", [])
        prompt_lower = prompt.lower()
        for concept in blocked:
            if concept.lower() in prompt_lower:
                return False, f"Prompt contains blocked concept: '{concept}'"
        return True, ""

    def _get_negative_prompt(self, custom: str = "") -> str:
        """Merge custom negative prompt with config defaults."""
        defaults = self.config.get("negative_prompts", {})
        base = defaults.get("family_friendly", defaults.get("default", ""))
        if custom:
            return f"{base}, {custom}"
        return base

    @torch.inference_mode()
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Full generation pipeline with safety guardrails.

        Steps:
            1. Screen prompt for blocked concepts
            2. Generate image with SDXL (+ optional refiner)
            3. Apply style conditioning if style_image provided
            4. Run post-generation safety evaluation
            5. Return result (filtered if unsafe)
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call .load() first.")

        # Step 1: Prompt screening
        prompt_safe, reason = self._screen_prompt(request.prompt)
        if not prompt_safe:
            logger.warning("Prompt blocked: %s", reason)
            blank = Image.new("RGB", (request.width, request.height), (0, 0, 0))
            return GenerationResult(
                image=blank,
                safety=SafetyResult(
                    is_safe=False,
                    nsfw_score=0.0,
                    content_rating=ContentRating.BLOCKED,
                    details=reason,
                ),
                prompt=request.prompt,
                seed=request.seed or 0,
                generation_time_s=0.0,
                was_filtered=True,
                filter_reason=reason,
            )

        # Step 2: Generate
        negative_prompt = self._get_negative_prompt(request.negative_prompt)
        seed = request.seed if request.seed is not None else torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        t0 = time.perf_counter()

        if request.use_refiner and self.refiner is not None:
            # Ensemble of expert denoisers
            latent = self.pipe(
                prompt=request.prompt,
                negative_prompt=negative_prompt,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                height=request.height,
                width=request.width,
                denoising_end=request.high_noise_frac,
                output_type="latent",
                generator=generator,
            ).images

            image = self.refiner(
                prompt=request.prompt,
                negative_prompt=negative_prompt,
                image=latent,
                num_inference_steps=request.num_inference_steps,
                denoising_start=request.high_noise_frac,
                generator=generator,
            ).images[0]
        else:
            image = self.pipe(
                prompt=request.prompt,
                negative_prompt=negative_prompt,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                height=request.height,
                width=request.width,
                generator=generator,
            ).images[0]

        gen_time = time.perf_counter() - t0

        # Step 3: Style conditioning (if requested)
        style_scores = {}
        if request.style_image_path:
            if not self._ip_adapter_loaded:
                st_cfg = self.config.get("style_transfer", {})
                load_ip_adapter(
                    self.pipe,
                    ip_adapter_model=st_cfg.get("ip_adapter_model", "h94/IP-Adapter"),
                    weight_name=st_cfg.get(
                        "ip_adapter_weight_name",
                        "ip-adapter-plus_sdxl_vit-h.safetensors",
                    ),
                    scale=request.ip_adapter_scale,
                )
                self._ip_adapter_loaded = True

            style_img = prepare_style_image(request.style_image_path)
            image = generate_with_style(
                self.pipe,
                prompt=request.prompt,
                style_image=style_img,
                negative_prompt=negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                ip_adapter_scale=request.ip_adapter_scale,
                seed=seed,
            )

        # Step 4: Safety evaluation
        safety_result = self.safety.evaluate(image, check_brand=bool(self._brand_refs))

        # Step 5: Filter if unsafe
        was_filtered = False
        filter_reason = ""
        if not safety_result.is_safe:
            was_filtered = True
            filter_reason = safety_result.details
            logger.warning("Image failed safety: %s — applying blur filter", filter_reason)
            image = image.filter(ImageFilter.GaussianBlur(radius=40))

        return GenerationResult(
            image=image,
            safety=safety_result,
            prompt=request.prompt,
            seed=seed,
            generation_time_s=gen_time,
            style_scores=style_scores,
            was_filtered=was_filtered,
            filter_reason=filter_reason,
        )

    def generate_batch(
        self,
        requests: list[GenerationRequest],
        output_dir: str = "sample_outputs",
    ) -> list[GenerationResult]:
        """Generate multiple images, saving safe outputs to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        for i, req in enumerate(requests):
            logger.info("Generating %d/%d: %s", i + 1, len(requests), req.prompt[:60])
            result = self.generate(req)
            results.append(result)

            if result.safety.is_safe:
                fname = f"gen_{i:03d}_seed{result.seed}.png"
                result.image.save(output_path / fname)
                logger.info(
                    "  ✓ Saved %s (rating=%s, time=%.1fs)",
                    fname,
                    result.safety.content_rating.value,
                    result.generation_time_s,
                )
            else:
                logger.warning(
                    "  ✗ Filtered: %s", result.filter_reason
                )

        safe_count = sum(1 for r in results if r.safety.is_safe)
        logger.info(
            "Batch complete: %d/%d passed safety", safe_count, len(results)
        )
        return results
