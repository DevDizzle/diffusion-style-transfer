# %% [markdown]
# # Style Transfer with IP-Adapter & Img2Img
#
# Apply artistic style conditioning to Stable Diffusion XL generations
# using IP-Adapter for image-prompted style transfer and img2img for
# style blending. Includes CLIP-based style similarity scoring.
#
# **Requirements:** GPU with ≥16GB VRAM. Run `generate.py` first.

# %% [markdown]
# ## 1. Setup

# %%
import sys
from pathlib import Path

import torch
from PIL import Image

project_root = Path("..").resolve()
sys.path.insert(0, str(project_root))

from src.style import (
    StyleSimilarityScorer,
    load_ip_adapter,
    prepare_style_image,
    generate_with_style,
)
from src.safety import SafetyEvaluator

print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 2. Load SDXL + IP-Adapter

# %%
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    solver_order=2,
    use_karras_sigmas=True,
)

# Load IP-Adapter for style conditioning
load_ip_adapter(
    pipe,
    ip_adapter_model="h94/IP-Adapter",
    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
    scale=0.6,
)

print("✓ SDXL + IP-Adapter loaded")

# %% [markdown]
# ## 3. Prepare Style Reference Images
#
# Replace these paths with your own style references.
# Good choices: concept art, storybook illustrations, animation stills.

# %%
# ── Style References ────────────────────────────────────────────────────────
# NOTE: Replace with actual paths to your reference images.
# For demonstration, we create synthetic reference images.

STYLE_REFS_DIR = project_root / "style_references"
STYLE_REFS_DIR.mkdir(exist_ok=True)

# Create placeholder style references (replace with real images)
def create_placeholder(path: Path, color: tuple, label: str):
    """Create a simple gradient placeholder image."""
    import numpy as np
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        for c in range(3):
            arr[y, :, c] = int(color[c] * (1 - y / 512) + 40 * (y / 512))
    img = Image.fromarray(arr)
    img.save(path)
    print(f"  Created placeholder: {path.name} ({label})")

# These would normally be curated reference images
styles = {
    "watercolor": ((180, 200, 230), "Soft watercolor style"),
    "cel_shaded": ((255, 200, 100), "Cel-shaded animation style"),
    "storybook": ((200, 180, 160), "Classic storybook illustration"),
}

for name, (color, label) in styles.items():
    path = STYLE_REFS_DIR / f"{name}_reference.png"
    if not path.exists():
        create_placeholder(path, color, label)

print("\n✓ Style references ready")
print("  ⚠ Replace placeholders with real reference images for best results!")

# %% [markdown]
# ## 4. Style-Conditioned Generation

# %%
PROMPT = (
    "A whimsical treehouse village in a magical forest, "
    "lanterns hanging from branches, warm evening light, "
    "highly detailed, vibrant colors, family-friendly"
)
NEGATIVE = (
    "ugly, blurry, low quality, deformed, dark, scary, "
    "nsfw, violence, watermark, text, signature"
)

# ── Generate with each style ────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# Baseline (no style conditioning)
pipe.set_ip_adapter_scale(0.0)  # disable IP-Adapter
baseline = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    num_inference_steps=40,
    guidance_scale=7.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
axes[0].imshow(baseline)
axes[0].set_title("Baseline (no style)", fontsize=12)
axes[0].axis("off")

# Style-conditioned variants
for i, (name, (_, label)) in enumerate(styles.items()):
    style_img = prepare_style_image(
        STYLE_REFS_DIR / f"{name}_reference.png",
        target_size=(1024, 1024),
    )
    result = generate_with_style(
        pipe,
        prompt=PROMPT,
        style_image=style_img,
        negative_prompt=NEGATIVE,
        num_inference_steps=40,
        guidance_scale=7.5,
        ip_adapter_scale=0.6,
        seed=42,
    )
    axes[i + 1].imshow(result)
    axes[i + 1].set_title(f"Style: {name}", fontsize=12)
    axes[i + 1].axis("off")

plt.suptitle("Style Transfer Comparison — Treehouse Village", fontsize=16)
plt.tight_layout()
plt.savefig(
    str(project_root / "sample_outputs" / "style_comparison.png"),
    dpi=150, bbox_inches="tight",
)
print("✓ Saved style_comparison.png")
plt.show()

# %% [markdown]
# ## 5. IP-Adapter Scale Sweep
#
# The scale parameter controls how strongly the style reference
# influences the output. Let's visualize the effect.

# %%
scales = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
style_img = prepare_style_image(
    STYLE_REFS_DIR / "watercolor_reference.png",
    target_size=(1024, 1024),
)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, scale in enumerate(scales):
    result = generate_with_style(
        pipe,
        prompt=PROMPT,
        style_image=style_img,
        negative_prompt=NEGATIVE,
        num_inference_steps=30,
        guidance_scale=7.5,
        ip_adapter_scale=scale,
        seed=42,
    )
    axes[i].imshow(result)
    axes[i].set_title(f"IP-Adapter scale={scale}", fontsize=12)
    axes[i].axis("off")

plt.suptitle("IP-Adapter Scale Sweep — Watercolor Style", fontsize=16)
plt.tight_layout()
plt.savefig(
    str(project_root / "sample_outputs" / "ip_adapter_scale_sweep.png"),
    dpi=150, bbox_inches="tight",
)
print("✓ Saved ip_adapter_scale_sweep.png")
plt.show()

# %% [markdown]
# ## 6. CLIP Style Similarity Scoring

# %%
scorer = StyleSimilarityScorer(device="cuda")
scorer.load()

# Register styles
for name in styles:
    scorer.register_style(
        name,
        [str(STYLE_REFS_DIR / f"{name}_reference.png")],
    )

# Also register text-based styles
scorer.register_style_from_text(
    "disney_classic",
    "Classic Disney animation style, hand-drawn, vibrant colors, "
    "expressive characters, magical atmosphere",
)
scorer.register_style_from_text(
    "pixar_3d",
    "Pixar 3D animation style, photorealistic rendering, "
    "warm lighting, appealing character design",
)

print("✓ Registered 5 styles for scoring")

# %%
# Score the baseline image against all styles
print("\nStyle Similarity Scores (Baseline Image):")
print("-" * 45)
for score in scorer.score_all(baseline):
    bar = "█" * int(score.similarity * 40)
    print(f"  {score.style_name:<18} {score.similarity:.4f}  {bar}")

best = scorer.best_match(baseline)
print(f"\n  Best match: {best.style_name} ({best.similarity:.4f})")

# %% [markdown]
# ## 7. Img2Img Style Transfer (Alternative Approach)
#
# Instead of IP-Adapter, use img2img to blend a style reference
# directly. This gives a different kind of style influence.

# %%
from diffusers import StableDiffusionXLImg2ImgPipeline

img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

# Use baseline image as init, with different strength values
strengths = [0.3, 0.5, 0.7, 0.9]

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

for i, strength in enumerate(strengths):
    result = img2img(
        prompt=PROMPT + ", watercolor painting style",
        negative_prompt=NEGATIVE,
        image=baseline.resize((1024, 1024)),
        strength=strength,
        guidance_scale=7.5,
        num_inference_steps=40,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    axes[i].imshow(result)
    axes[i].set_title(f"img2img strength={strength}", fontsize=12)
    axes[i].axis("off")

plt.suptitle("Img2Img Style Transfer — Strength Comparison", fontsize=16)
plt.tight_layout()
plt.savefig(
    str(project_root / "sample_outputs" / "img2img_strength.png"),
    dpi=150, bbox_inches="tight",
)
print("✓ Saved img2img_strength.png")
plt.show()

# %% [markdown]
# ## 8. Safety Check on Style-Transferred Outputs

# %%
safety = SafetyEvaluator(
    device="cuda",
    nsfw_threshold=0.85,
    brand_threshold=0.25,
    blocked_concepts=["violence", "gore", "explicit"],
)
safety.load_all()

# Check baseline
result = safety.evaluate(baseline, check_brand=False)
print(f"Baseline safety: safe={result.is_safe} | {result.details}")

# %% [markdown]
# ---
# **Next steps:**
# - Replace placeholder style references with real curated images
# - Fine-tune IP-Adapter scale per style for best results
# - See `GUIDE.md` for DreamBooth fine-tuning as a stretch goal
