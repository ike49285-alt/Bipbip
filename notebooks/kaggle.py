"""Run the image steps on a free Kaggle GPU instead of your own card.

Paste these cells into a Kaggle notebook with the GPU accelerator on. There is
no separate pipeline here -- it clones the repo and runs the same `python -m
bot` commands you would run locally. One code path, two places to run it.

Why bother when you have a GPU: Kaggle gives you a 16GB T4 or P100, which fits
SDXL at 1024px. A 6GB card is limited to SD 1.5 at 512px. The pictures are
better, and it costs nothing but the round trip.

What it costs you: this is batch work. You upload the anchor, run, and download
the results. Sessions are capped (about 9 hours each, 30 GPU-hours a week) and
disconnect, though `generate` resumes from whatever is already on disk.
"""

# %% 1. the repo -----------------------------------------------------------
# !git clone -q --branch claude/repo-cleanup-9x3t2e https://github.com/ike49285-alt/Bipbip /kaggle/working/Bipbip
# !pip install -q "diffusers==0.31.0" transformers accelerate safetensors peft bitsandbytes

# %% 2. the anchor ---------------------------------------------------------
# The anchor is gitignored, so a fresh clone does not have it. Add it as a
# private Kaggle Dataset via "+ Add Input" -- this finds it wherever it landed,
# so the dataset name and filename do not have to match anything.
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/kaggle/working/Bipbip")
INPUTS = Path("/kaggle/input")
OUT = Path("/kaggle/working/out")

sys.path.insert(0, str(REPO))


def find_anchor():
    found = sorted(p for p in INPUTS.glob("*/**/*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if len(found) == 1:
        return found[0]
    if not found:
        existing = [d.name for d in INPUTS.iterdir()] if INPUTS.exists() else []
        raise SystemExit(
            "No image found under /kaggle/input.\n"
            f"Datasets attached: {existing or 'none'}\n"
            "Use '+ Add Input' in the right-hand panel to attach the dataset "
            "holding your anchor image."
        )
    raise SystemExit(
        "Several images are attached; pick one explicitly:\n"
        + "\n".join(f"  {p}" for p in found[:10])
        + "\n\nSet: anchor_source = Path(\"<one of the above>\")"
    )


anchor_source = find_anchor()
anchor_target = REPO / "content/anchor/remy-anchor.jpg"
anchor_target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(anchor_source, anchor_target)
print(f"anchor: {anchor_source}  ->  {anchor_target}")


def bot(*args):
    """Run a bot command in the repo, streaming its output."""
    subprocess.run([sys.executable, "-m", "bot", *args], cwd=REPO, check=True)


bot("doctor")

# %% 3. pick the anchor strength ------------------------------------------
# Four renders of one prompt. Look at them before spending a session on 200.
bot("generate", "--sweep", "--out", str(OUT), "--family", "sdxl")

# %% 4. the batch ----------------------------------------------------------
# Set --scale to whichever sweep image you liked. Resumes if the session drops.
bot("generate", "-n", "200", "--scale", "0.65", "--out", str(OUT), "--family", "sdxl")

# %% 5. curate -------------------------------------------------------------
bot("curate", "--keep", "40", "--out", str(OUT))

# %% 6. train --------------------------------------------------------------
# !git clone -q --depth 1 --branch v0.31.0 https://github.com/huggingface/diffusers /kaggle/working/diffusers
bot("train", "--out", str(OUT), "--family", "sdxl",
    "--lora-out", "/kaggle/working/lora",
    "--trainer", "/kaggle/working/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py")

# %% 7. take it home -------------------------------------------------------
# The LoRA is the thing worth keeping -- a few hundred MB, versus gigabytes of
# bootstrap images you will never look at again.
shutil.make_archive("/kaggle/working/remy-lora", "zip", "/kaggle/working/lora")
print("download remy-lora.zip from the output panel, unzip into content/lora/,")
print("then locally:  python -m bot generate --lora content/lora/<file>.safetensors")
