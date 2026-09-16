"""Kaggle side, part 2: train the character LoRA on the curated set.

This does not implement training. It fetches the maintained diffusers trainer
at a pinned version and invokes it with flags sized for a free 16GB GPU --
writing a training loop by hand would be strictly worse than the one that is
maintained. The command is built by `bot.bootstrap.train_command`, which is
covered by tests, so a mistyped flag is caught before it costs a GPU hour.

Run `notebooks/bootstrap.py` first; this expects its curated output.
"""

# %% install ---------------------------------------------------------------
# !pip install -q "diffusers==0.31.0" transformers accelerate safetensors peft bitsandbytes
# !git clone --depth 1 --branch v0.31.0 https://github.com/huggingface/diffusers /kaggle/working/diffusers
# !pip install -q -r /kaggle/working/diffusers/examples/dreambooth/requirements_sdxl.txt

# %% config ----------------------------------------------------------------
import subprocess
import sys
from pathlib import Path

REPO = Path("/kaggle/working/Bipbip")
TRAIN_DIR = Path("/kaggle/working/out/train")
OUTPUT_DIR = Path("/kaggle/working/lora")
TRAINER = Path("/kaggle/working/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py")

# A rare token the base model has no prior for. Do not use her name -- "remy"
# already means things to the model, and the LoRA will fight that.
INSTANCE_TOKEN = "rmyx"
STEPS = 1200        # ~30 steps per image for a 40-image set
RANK = 16           # identity, not style; raise only if likeness stays soft

sys.path.insert(0, str(REPO))
from bot.bootstrap import train_command  # noqa: E402

images = sorted(p for p in TRAIN_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
assert images, f"no training images in {TRAIN_DIR} -- run bootstrap.py first"
assert TRAINER.exists(), f"trainer not found at {TRAINER} -- run the clone cell above"
print(f"{len(images)} images, ~{STEPS // max(len(images), 1)} steps each")

# %% train -----------------------------------------------------------------
cmd = train_command(
    TRAIN_DIR, OUTPUT_DIR,
    instance_token=INSTANCE_TOKEN, trainer=str(TRAINER), steps=STEPS, rank=RANK,
)
print(" \\\n  ".join(cmd))
subprocess.run(cmd, check=True)

# %% verify ----------------------------------------------------------------
weights = list(OUTPUT_DIR.glob("*.safetensors"))
assert weights, f"training produced no weights in {OUTPUT_DIR}"
print(f"trained: {weights[0]}  ({weights[0].stat().st_size / 1e6:.1f} MB)")
print(f"copy it to {REPO / 'content/lora/'} and set visual.lora_path")

# %% smoke test ------------------------------------------------------------
# Generate one image with the LoRA loaded and run it through the same QC gate.
# If the score is not clearly better than the bootstrap average, the LoRA did
# not take -- check the sweep scale and the curated set before spending more.
#
# import torch
# from diffusers import AutoPipelineForText2Image
# from bot.persona import Persona
# from bot.qc import ClipEmbedder, QCGate
#
# persona = Persona.load(REPO / "persona.json")
# pipe = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     torch_dtype=torch.float16, variant="fp16",
# ).to("cuda")
# pipe.load_lora_weights(str(weights[0]))
# image = pipe(prompt=f"a picture of {INSTANCE_TOKEN}, cluttered bedroom, "
#                     "digital illustration, anime style, cel shading").images[0]
# image.save("/kaggle/working/lora-smoke.png")
#
# embedder = ClipEmbedder()
# gate = QCGate(embedder.embed(REPO / persona.visual.anchor_image),
#               persona.visual.identity_threshold)
# print(gate.judge(Path("/kaggle/working/lora-smoke.png"),
#                  embedder.embed(Path("/kaggle/working/lora-smoke.png"))))
