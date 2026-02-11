# 🎨 Diffusion Style Transfer

**Production-grade text-to-image generation with style conditioning and content safety guardrails.**

Built with Stable Diffusion XL, IP-Adapter, and CLIP — designed for creative pipelines where brand consistency and content safety are non-negotiable.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Generation Pipeline                       │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Prompt   │───▶│ Safety Gate  │───▶│   SDXL Base      │  │
│  │  Input    │    │ (pre-screen) │    │   + Refiner      │  │
│  └──────────┘    └──────────────┘    │   + IP-Adapter    │  │
│                                      └────────┬─────────┘  │
│                                               │             │
│  ┌──────────────────────────────────────────────┐           │
│  │            Post-Generation Safety            │           │
│  │  ┌────────────┐ ┌───────────┐ ┌───────────┐ │           │
│  │  │   NSFW     │ │  Content  │ │   Brand   │ │           │
│  │  │ Classifier │ │  Rating   │ │Consistency│ │           │
│  │  │(diffusers) │ │(G/PG/PG13)│ │  (CLIP)   │ │           │
│  │  └────────────┘ └───────────┘ └───────────┘ │           │
│  └──────────────────────────┬───────────────────┘           │
│                             │                               │
│                    ┌────────▼────────┐                      │
│                    │  Safe Output /  │                      │
│                    │  Blur+Block     │                      │
│                    └─────────────────┘                      │
└─────────────────────────────────────────────────────────────┘

Style Conditioning:
  Reference Image ──▶ IP-Adapter ──▶ Style-Conditioned Latents
  Reference Image ──▶ CLIP Encoder ──▶ Style Similarity Score
```

---

## Features

| Feature | Description |
|---------|-------------|
| **SDXL Base + Refiner** | Ensemble of expert denoisers for maximum quality |
| **IP-Adapter Style Transfer** | Image-prompted style conditioning with tunable strength |
| **NSFW Detection** | Diffusers safety_checker + CLIP zero-shot fallback |
| **Content Rating** | Automated G / PG / PG-13 classification via CLIP |
| **Brand Consistency** | CLIP cosine similarity to reference brand images |
| **Prompt Safety Screening** | Pre-generation blocked concept filtering |
| **Batch Generation** | Generate + evaluate + save with full audit trail |

---

## Tech Stack

- **[Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)** — State-of-the-art text-to-image
- **[HuggingFace Diffusers](https://github.com/huggingface/diffusers)** — Pipeline framework
- **[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)** — Image-prompted style conditioning
- **[OpenCLIP ViT-H-14](https://github.com/mlfoundations/open_clip)** — Style similarity & content classification
- **PyTorch 2.2+** with CUDA / float16

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/diffusion-style-transfer.git
cd diffusion-style-transfer
pip install -r requirements.txt

# Run generation notebook
cd notebooks
python generate.py          # or open as Jupyter notebook

# Run style transfer
python style_transfer.py
```

See **[GUIDE.md](GUIDE.md)** for detailed setup instructions.

---

## Project Structure

```
diffusion-style-transfer/
├── README.md                  # This file
├── GUIDE.md                   # Step-by-step execution guide
├── requirements.txt           # Python dependencies
├── configs/
│   └── model_config.yaml      # Model, scheduler, safety thresholds
├── notebooks/
│   ├── generate.py            # Text-to-image generation (percent script)
│   └── style_transfer.py      # Style transfer & scoring (percent script)
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # End-to-end generation pipeline
│   ├── safety.py              # NSFW, content rating, brand consistency
│   └── style.py               # Style encoding, similarity, IP-Adapter utils
└── sample_outputs/            # Generated images (git-ignored)
    └── .gitkeep
```

---

## Sample Outputs

> *Run the notebooks on a GPU to generate these.*

| Prompt | Rating | Style |
|--------|--------|-------|
| Enchanted castle at golden hour | G | Baseline |
| Underwater kingdom with coral palaces | G | Watercolor |
| Woodland creatures tea party | G | Storybook |
| Young astronaut discovering alien garden | G | Cel-shaded |

---

## Content Safety Design

This project treats content safety as a **first-class concern**, not an afterthought:

1. **Pre-generation:** Prompts are screened against a configurable blocklist before any GPU time is spent
2. **Post-generation:** Every image passes through NSFW detection, content rating, and (optionally) brand consistency scoring
3. **Fail-safe:** Unsafe images are automatically blurred and flagged — never silently passed through
4. **Configurable thresholds:** All safety parameters are in `configs/model_config.yaml`
5. **Audit trail:** Every `GenerationResult` includes full safety metadata

This approach is directly applicable to entertainment and media production pipelines where brand integrity and audience-appropriate content are paramount.

---

## License

MIT

---

*Built as a portfolio demonstration of production AI image generation with safety-first design.*
