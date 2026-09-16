"""Step 3b, part 1: generate the bootstrap set on a free Kaggle GPU.

Paste into a Kaggle notebook (GPU accelerator on) or run with `# %%` cells in
an IDE. Reads a manifest from `python -m bot.scenes --json`, generates each
shot conditioned on the anchor via IP-Adapter, scores everything through the
QC gate, and writes the survivors plus a report.

WHY IP-ADAPTER PLUS AND NOT FACEID: the FaceID variants extract InsightFace
face embeddings, which are trained on photographs and routinely fail to detect
an illustrated face at all. The Plus variants use a CLIP vision encoder and
handle illustration fine. Same reason the QC gate uses CLIP.

UNVERIFIED: this has not been executed -- no GPU in the authoring environment.
Expect to fix at least one call signature on the first run; diffusers moves.
"""

# %% install ---------------------------------------------------------------
# !pip install -q diffusers transformers accelerate safetensors

# %% config ----------------------------------------------------------------
import json
from pathlib import Path

REPO = Path("/kaggle/working/Bipbip")       # where you cloned this repo
ANCHOR = REPO / "content/anchor/remy-anchor.jpg"
MANIFEST = REPO / "content/manifest.json"    # from: python -m bot.scenes -n 200 --json
OUT = Path("/kaggle/working/out")

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
# plus-face crops tighter on identity; plus conditions on the whole image.
# Both are CLIP-based. Start with plus-face and fall back if framing suffers.
IP_ADAPTER_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.bin"

# The single most important knob. Too high and every output apes the anchor's
# pose and framing; too low and identity drifts between scenes. Sweep this
# before committing to a 200-image run.
IP_ADAPTER_SCALE = 0.65

STEPS = 30
GUIDANCE = 7.0
SIZE = 1024

# %% load ------------------------------------------------------------------
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

OUT.mkdir(parents=True, exist_ok=True)
shots = json.loads(MANIFEST.read_text())
anchor = Image.open(ANCHOR).convert("RGB")
print(f"{len(shots)} shots queued, anchor {anchor.size}")

pipe = AutoPipelineForText2Image.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
pipe.load_ip_adapter(
    IP_ADAPTER_REPO,
    subfolder=IP_ADAPTER_SUBFOLDER,
    weight_name=IP_ADAPTER_WEIGHT,
    low_cpu_mem_usage=True,
)
pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
pipe.set_progress_bar_config(disable=True)

# %% scale sweep -- run this FIRST, on one prompt, before the full batch ----
def sweep(shot, scales=(0.4, 0.55, 0.7, 0.85)):
    """Generate one prompt at several scales so you can eyeball the tradeoff."""
    frames = []
    for s in scales:
        pipe.set_ip_adapter_scale(s)
        img = pipe(
            prompt=shot["prompt"],
            negative_prompt=shot["negative"],
            ip_adapter_image=anchor,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            height=SIZE, width=SIZE,
            generator=torch.Generator("cuda").manual_seed(shot["seed"]),
        ).images[0]
        img.save(OUT / f"sweep-{s}.png")
        frames.append((s, img))
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
    return frames

# sweep(shots[0])   # <- uncomment, look at the four, then set IP_ADAPTER_SCALE

# %% generate --------------------------------------------------------------
def generate(shot):
    out = OUT / f"shot-{shot['index']:04d}.png"
    if out.exists():                      # resume after a session timeout
        return out
    image = pipe(
        prompt=shot["prompt"],
        negative_prompt=shot["negative"],
        ip_adapter_image=anchor,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        height=SIZE, width=SIZE,
        generator=torch.Generator("cuda").manual_seed(shot["seed"]),
    ).images[0]
    image.save(out)
    return out

paths = []
for n, shot in enumerate(shots, 1):
    paths.append(generate(shot))
    if n % 10 == 0:
        print(f"{n}/{len(shots)}")

# %% qc --------------------------------------------------------------------
import sys
sys.path.insert(0, str(REPO))
from bot.persona import Persona
from bot.qc import ClipEmbedder, QCGate, curate, summarise

persona = Persona.load(REPO / "persona.json")
assert persona.visual.identity_check == "clip", "face embedders fail on illustration"

embedder = ClipEmbedder()
anchor_vec = embedder.embed(ANCHOR)
gate = QCGate(anchor_vec, persona.visual.identity_threshold)

vectors = {p: embedder.embed(p) for p in paths}
judgements = [gate.judge(p, v) for p, v in vectors.items()]
print(summarise(judgements))

# %% curate ----------------------------------------------------------------
KEEP = 40   # LoRA training set size

picked = curate(judgements, vectors, keep=KEEP)
print(f"curated {len(picked)} for training")
for j in picked[:10]:
    print(" ", j)

train_dir = OUT / "train"
train_dir.mkdir(exist_ok=True)
for j in picked:
    Image.open(j.image).save(train_dir / j.image.name)

(OUT / "qc-report.json").write_text(json.dumps(
    [{"image": str(j.image), "score": j.score, "passed": j.passed} for j in judgements],
    indent=2,
))
print(f"training set at {train_dir} -- REVIEW IT BY EYE before training")

# %% ------------------------------------------------------------------------
# Next: LoRA training on train_dir. Left deliberately unwritten -- the trainer
# APIs (kohya, diffusers' train_text_to_image_lora_sdxl.py) churn faster than
# anything above, and writing it blind would cost you a session debugging my
# guesses. Pin a trainer version first, then we script it.
