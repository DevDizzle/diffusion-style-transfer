# 📋 Execution Guide

Step-by-step instructions to run this project end-to-end and capture portfolio-ready outputs.

**Estimated time: 4–6 hours** (including model downloads and generation)

---

## Prerequisites

- Python 3.10+
- NVIDIA GPU with ≥16GB VRAM (A100, A10G, RTX 4090, or similar)
- ~25GB disk space for model weights
- HuggingFace account (free) for model access

---

## Step 1: Get a GPU Runtime (30 min)

**Option A — Google Colab Pro ($10/mo)**
```
Runtime → Change runtime type → T4 or A100
```

**Option B — Lambda Labs / RunPod / Vast.ai**
```bash
# A100 80GB ~$1.10/hr, A10G ~$0.60/hr
# SSH in and clone the repo
```

**Option C — Local GPU**
```bash
# RTX 4090 (24GB) or RTX 3090 (24GB) works well
nvidia-smi  # verify CUDA is available
```

---

## Step 2: Install Dependencies & Download Models (45 min)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/diffusion-style-transfer.git
cd diffusion-style-transfer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pre-download models (optional — they auto-download on first run)
python -c "
from diffusers import StableDiffusionXLPipeline
StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype='auto',
    variant='fp16',
)
print('✓ SDXL base downloaded')
"
```

> **Note:** First download is ~6.5GB for the base model + ~6GB for the refiner. Models are cached in `~/.cache/huggingface/`.

---

## Step 3: Run the Generate Notebook (1–2 hours)

```bash
cd notebooks

# Option A: Run as a script
python generate.py

# Option B: Open as Jupyter notebook
pip install jupytext jupyter
jupytext --to notebook generate.py
jupyter notebook generate.ipynb
```

**What this does:**
1. Loads SDXL base + refiner
2. Generates images from 4 curated prompts
3. Runs guidance scale comparison grid
4. Runs scheduler comparison grid
5. Batch generates with safety checks
6. Saves all outputs to `sample_outputs/`

**Expected outputs:**
- `sample_outputs/enchanted_castle.png`
- `sample_outputs/guidance_comparison.png`
- `sample_outputs/scheduler_comparison.png`
- `sample_outputs/gen_000_seed42.png` through `gen_003_seed42.png`

---

## Step 4: Run the Style Transfer Notebook (1–2 hours)

```bash
python style_transfer.py
```

**Before running:** Replace the placeholder style references in `style_references/` with real images:
- A watercolor painting
- A cel-shaded animation frame
- A classic storybook illustration

**What this does:**
1. Loads SDXL + IP-Adapter
2. Generates baseline + 3 style-conditioned variants
3. Sweeps IP-Adapter scale (0.0 → 1.0)
4. Scores all images for style similarity via CLIP
5. Demonstrates img2img style transfer alternative
6. Runs safety checks on all outputs

**Expected outputs:**
- `sample_outputs/style_comparison.png`
- `sample_outputs/ip_adapter_scale_sweep.png`
- `sample_outputs/img2img_strength.png`

---

## Step 5: Capture Sample Outputs (30 min)

1. Review all images in `sample_outputs/`
2. Pick the 4–6 best results
3. Rename them clearly (e.g., `hero_enchanted_castle.png`)
4. Add them to the repo (update `.gitignore` to allow selected outputs)

```bash
# Allow specific sample outputs
echo '!sample_outputs/hero_*.png' >> .gitignore
git add sample_outputs/hero_*.png
```

---

## Step 6: Update README (30 min)

1. Replace the sample outputs table with actual images:
```markdown
![Enchanted Castle](sample_outputs/hero_enchanted_castle.png)
```
2. Update any metrics (generation times, safety scores)
3. Add your GitHub username to the clone URL
4. Push to GitHub

```bash
git add -A
git commit -m "Add generated sample outputs and finalize README"
git push origin main
```

---

## 🌟 Stretch Goal: DreamBooth Fine-Tuning

Fine-tune SDXL on a specific visual style using DreamBooth for even stronger brand consistency.

**Estimated additional time: 3–5 hours**

### Setup

```bash
pip install peft bitsandbytes
```

### Prepare Training Data

Collect 15–30 images in your target style. Place them in `data/style_training/`.

### Fine-Tune with LoRA

```python
# dreambooth_finetune.py (create this file)
from diffusers import StableDiffusionXLPipeline
from peft import LoraConfig

# LoRA config for memory-efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_v", "to_k", "to_out.0"],
    lora_dropout=0.05,
)

# Use accelerate for distributed training
# accelerate launch dreambooth_finetune.py \
#   --pretrained_model="stabilityai/stable-diffusion-xl-base-1.0" \
#   --instance_data_dir="data/style_training" \
#   --instance_prompt="a painting in the style of sks" \
#   --output_dir="checkpoints/dreambooth_style" \
#   --resolution=1024 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --learning_rate=1e-4 \
#   --lr_scheduler="cosine" \
#   --max_train_steps=1000 \
#   --use_lora \
#   --rank=16
```

### Using the HuggingFace Training Script

```bash
accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="data/style_training" \
  --instance_prompt="an illustration in the style of sks" \
  --output_dir="checkpoints/dreambooth_lora" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --max_train_steps=1000 \
  --validation_prompt="a castle in the style of sks" \
  --validation_epochs=50 \
  --seed=42
```

### Load Fine-Tuned Weights

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# Load LoRA weights
pipe.load_lora_weights("checkpoints/dreambooth_lora")

# Generate with the fine-tuned style
image = pipe("a forest in the style of sks").images[0]
```

### Why DreamBooth Matters for Brand Consistency

- **IP-Adapter** gives approximate style transfer from reference images
- **DreamBooth LoRA** learns the actual visual distribution of a brand's style
- Combined with the safety pipeline, this enables **on-brand, content-safe generation at scale**

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Use `torch.float16`, reduce resolution to 768×768, or enable `pipe.enable_model_cpu_offload()` |
| Model download fails | Check HuggingFace token: `huggingface-cli login` |
| IP-Adapter not found | Ensure `ip-adapter` package is installed and model repo is accessible |
| Safety checker false positives | Adjust thresholds in `configs/model_config.yaml` |
| Slow generation | Enable `torch.compile(pipe.unet)` for 20-40% speedup on Ampere+ GPUs |

---

*Total estimated time: 4–6 hours for core, +3–5 hours for DreamBooth stretch goal.*
