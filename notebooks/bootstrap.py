"""Kaggle side, part 1: generate the bootstrap set from the anchor.

Thin on purpose. Everything except the diffusers calls lives in
`bot/bootstrap.py`, where it is covered by tests that run without a GPU --
resume, failure tolerance, gating, curation, the report. What is left here is
the part that genuinely needs the hardware.

    Kaggle: GPU accelerator ON, internet ON.

WHY IP-ADAPTER PLUS, NOT FACEID: the FaceID variants extract InsightFace face
embeddings, which are trained on photographs and routinely fail to detect an
illustrated face at all. Plus uses a CLIP vision encoder. Same reason the QC
gate uses CLIP.
"""

# %% install ---------------------------------------------------------------
# !pip install -q "diffusers==0.31.0" transformers accelerate safetensors peft

# %% config ----------------------------------------------------------------
import json
import sys
from pathlib import Path

REPO = Path("/kaggle/working/Bipbip")
OUT = Path("/kaggle/working/out")
MANIFEST = REPO / "content/manifest.json"     # python -m bot prompts -n 200 --json

IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.bin"

# The knob that decides whether any of this works. Too high and every output
# copies the anchor's pose and framing; too low and identity drifts between
# scenes. Sweep it before committing to a full batch.
IP_ADAPTER_SCALE = 0.65
STEPS, GUIDANCE, SIZE = 30, 7.0, 1024
KEEP = 40                                     # LoRA training set size

sys.path.insert(0, str(REPO))
from bot.bootstrap import (                   # noqa: E402
    assemble_training_set, gate_images, run_batch, write_report,
)
from bot.persona import Persona               # noqa: E402
from bot.qc import ClipEmbedder, summarise    # noqa: E402

persona = Persona.load(REPO / "persona.json")
ANCHOR = REPO / persona.visual.anchor_image
assert ANCHOR.exists(), f"anchor missing at {ANCHOR} -- upload it as a Kaggle dataset"
assert persona.visual.identity_check == "clip", "face embedders fail on illustration"

shots = json.loads(MANIFEST.read_text())
print(f"{len(shots)} shots queued")

# %% pipeline --------------------------------------------------------------
import torch                                  # noqa: E402
from diffusers import AutoPipelineForText2Image  # noqa: E402
from PIL import Image                         # noqa: E402

anchor = Image.open(ANCHOR).convert("RGB")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
).to("cuda")
pipe.load_ip_adapter(
    IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER,
    weight_name=IP_ADAPTER_WEIGHT, low_cpu_mem_usage=True,
)
pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
pipe.set_progress_bar_config(disable=True)


def render(shot, scale=None):
    if scale is not None:
        pipe.set_ip_adapter_scale(scale)
    return pipe(
        prompt=shot["prompt"], negative_prompt=shot["negative"],
        ip_adapter_image=anchor, num_inference_steps=STEPS,
        guidance_scale=GUIDANCE, height=SIZE, width=SIZE,
        generator=torch.Generator("cuda").manual_seed(int(shot["seed"])),
    ).images[0]


def generate(shot, target):
    """The one callable bot.bootstrap needs from this notebook."""
    render(shot).save(target)


# %% scale sweep -- DO THIS FIRST, on one prompt ---------------------------
def sweep(shot, scales=(0.4, 0.55, 0.7, 0.85)):
    OUT.mkdir(parents=True, exist_ok=True)
    for s in scales:
        render(shot, scale=s).save(OUT / f"sweep-{s}.png")
        print(f"wrote sweep-{s}.png")
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)

# sweep(shots[0])   # <- uncomment, look at the four, then set IP_ADAPTER_SCALE

# %% generate --------------------------------------------------------------
result = run_batch(
    shots, generate, OUT,
    on_progress=lambda n, total: print(f"{n}/{total}") if n % 10 == 0 else None,
)
print(result.summary())
for index, reason in result.failed:
    print(f"  shot {index}: {reason}")

# %% gate and curate -------------------------------------------------------
embedder = ClipEmbedder()
judgements, vectors = gate_images(
    result.images, ANCHOR, embedder, persona.visual.identity_threshold
)
print(summarise(judgements))

picked = assemble_training_set(judgements, vectors, OUT / "train", keep=KEEP)
write_report(judgements, picked, OUT / "qc-report.json")
print(f"{len(picked)} curated into {OUT / 'train'}")
print("REVIEW THAT FOLDER BY EYE before training -- the gate catches off-model,")
print("not ugly.")
