# %% [markdown]
# # Text-to-Image Generation with Stable Diffusion XL
#
# This notebook demonstrates text-to-image generation using HuggingFace
# Diffusers with SDXL, including prompt engineering, negative prompts,
# guidance scale tuning, and batch generation with safety guardrails.
#
# **Requirements:** GPU with ≥16GB VRAM (A100/A10G recommended)

# %% [markdown]
# ## 1. Setup & Imports

# %%
import sys
from pathlib import Path

import torch
from PIL import Image

# Add project root to path
project_root = Path("..").resolve()
sys.path.insert(0, str(project_root))

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Load SDXL Pipeline

# %%
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Load base model
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

# Configure scheduler for faster, high-quality sampling
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    solver_order=2,
    use_karras_sigmas=True,
)

# Load refiner for ensemble of expert denoisers
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    REFINER_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

print("✓ SDXL base + refiner loaded")

# %% [markdown]
# ## 3. Prompt Engineering
#
# Good prompts for SDXL follow this structure:
# `[subject], [style/medium], [details], [lighting], [quality modifiers]`

# %%
# ── Prompt Library ──────────────────────────────────────────────────────────

PROMPTS = {
    "enchanted_castle": {
        "prompt": (
            "A majestic enchanted castle at golden hour, "
            "surrounded by lush gardens with glowing fireflies, "
            "digital painting, highly detailed, cinematic lighting, "
            "vibrant colors, fantasy art, 8k resolution"
        ),
        "negative_prompt": (
            "ugly, blurry, low quality, deformed, dark, scary, "
            "nsfw, violence, watermark, text, signature"
        ),
    },
    "underwater_kingdom": {
        "prompt": (
            "An underwater kingdom with crystal coral palaces, "
            "bioluminescent sea creatures, rays of sunlight filtering "
            "through clear blue water, concept art, artstation trending, "
            "highly detailed, vivid colors, family-friendly"
        ),
        "negative_prompt": (
            "ugly, blurry, low quality, deformed, murky, dark, "
            "nsfw, scary, realistic gore, watermark, text"
        ),
    },
    "forest_friends": {
        "prompt": (
            "Adorable woodland creatures having a tea party in a "
            "sun-dappled forest clearing, mushroom houses in background, "
            "storybook illustration style, soft watercolor textures, "
            "warm lighting, whimsical, charming, highly detailed"
        ),
        "negative_prompt": (
            "ugly, blurry, low quality, deformed, dark, scary, "
            "nsfw, violence, realistic, photographic, watermark"
        ),
    },
    "space_explorer": {
        "prompt": (
            "A brave young astronaut discovering a garden growing on "
            "a distant planet, colorful alien flowers, two moons in the sky, "
            "digital art, Pixar-inspired, warm and hopeful atmosphere, "
            "cinematic composition, 4k, family-friendly"
        ),
        "negative_prompt": (
            "ugly, blurry, low quality, deformed, dark, horror, "
            "nsfw, violence, scary aliens, watermark, text"
        ),
    },
}

print(f"Loaded {len(PROMPTS)} prompt templates")

# %% [markdown]
# ## 4. Single Image Generation

# %%
def generate_with_refiner(
    prompt: str,
    negative_prompt: str = "",
    seed: int = 42,
    guidance_scale: float = 7.5,
    num_steps: int = 40,
    high_noise_frac: float = 0.8,
) -> Image.Image:
    """Generate with SDXL base + refiner ensemble."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Base model: denoise from pure noise to high_noise_frac
    latent = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
        generator=generator,
    ).images

    # Refiner: denoise the remaining steps
    image = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=latent,
        num_inference_steps=num_steps,
        denoising_start=high_noise_frac,
        generator=generator,
    ).images[0]

    return image


# Generate the enchanted castle
prompt_data = PROMPTS["enchanted_castle"]
image = generate_with_refiner(
    prompt=prompt_data["prompt"],
    negative_prompt=prompt_data["negative_prompt"],
    seed=42,
)
image.save("../sample_outputs/enchanted_castle.png")
print("✓ Saved enchanted_castle.png")
image  # Display in notebook

# %% [markdown]
# ## 5. Guidance Scale Comparison
#
# Guidance scale controls how closely the image follows the prompt.
# Low values (2-5) = more creative/abstract.
# High values (10-20) = more literal but can oversaturate.

# %%
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt

guidance_values = [3.0, 5.0, 7.5, 10.0, 12.0, 15.0]
prompt_data = PROMPTS["forest_friends"]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, gs in enumerate(guidance_values):
    img = generate_with_refiner(
        prompt=prompt_data["prompt"],
        negative_prompt=prompt_data["negative_prompt"],
        seed=42,
        guidance_scale=gs,
        num_steps=30,  # fewer steps for comparison grid
    )
    axes[i].imshow(img)
    axes[i].set_title(f"guidance_scale={gs}", fontsize=14)
    axes[i].axis("off")

plt.suptitle("Guidance Scale Comparison — Forest Friends", fontsize=16)
plt.tight_layout()
plt.savefig("../sample_outputs/guidance_comparison.png", dpi=150, bbox_inches="tight")
print("✓ Saved guidance_comparison.png")
plt.show()

# %% [markdown]
# ## 6. Batch Generation with Safety Checks
#
# Using our pipeline module for production-grade generation with
# integrated content safety and brand consistency scoring.

# %%
from src.pipeline import DiffusionPipeline, GenerationRequest

pipeline = DiffusionPipeline(
    config_path=str(project_root / "configs" / "model_config.yaml"),
    device="cuda",
)
pipeline.load()

# %%
# Build batch requests from our prompt library
requests = []
for name, data in PROMPTS.items():
    requests.append(
        GenerationRequest(
            prompt=data["prompt"],
            negative_prompt=data["negative_prompt"],
            seed=42,
            use_refiner=True,
        )
    )

results = pipeline.generate_batch(
    requests,
    output_dir=str(project_root / "sample_outputs"),
)

# %%
# Summarize results
print("\n" + "=" * 60)
print("BATCH GENERATION SUMMARY")
print("=" * 60)
for i, (name, result) in enumerate(zip(PROMPTS.keys(), results)):
    status = "✓ PASS" if result.safety.is_safe else "✗ FILTERED"
    print(
        f"  {status} | {name:<25} | "
        f"rating={result.safety.content_rating.value:<5} | "
        f"nsfw={result.safety.nsfw_score:.3f} | "
        f"time={result.generation_time_s:.1f}s"
    )
print("=" * 60)

# %% [markdown]
# ## 7. Scheduler Comparison (Optional)
#
# Different schedulers trade off speed vs. quality.

# %%
from diffusers import (
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)

schedulers = {
    "DPM++ 2M Karras": DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    ),
    "Euler": EulerDiscreteScheduler.from_config(pipe.scheduler.config),
    "Euler Ancestral": EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    ),
    "DDIM": DDIMScheduler.from_config(pipe.scheduler.config),
}

prompt_data = PROMPTS["space_explorer"]

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
for i, (sched_name, scheduler) in enumerate(schedulers.items()):
    pipe.scheduler = scheduler
    img = pipe(
        prompt=prompt_data["prompt"],
        negative_prompt=prompt_data["negative_prompt"],
        guidance_scale=7.5,
        num_inference_steps=30,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    axes[i].imshow(img)
    axes[i].set_title(sched_name, fontsize=12)
    axes[i].axis("off")

plt.suptitle("Scheduler Comparison — Space Explorer", fontsize=16)
plt.tight_layout()
plt.savefig("../sample_outputs/scheduler_comparison.png", dpi=150, bbox_inches="tight")
print("✓ Saved scheduler_comparison.png")
plt.show()

# %% [markdown]
# ---
# **Next:** Open `style_transfer.py` to apply artistic style conditioning
# using IP-Adapter.
