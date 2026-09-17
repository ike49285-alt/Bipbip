"""Generating her pictures on your own GPU.

Sized for a 6GB card, which is the constraint that picks everything else:
SD 1.5 rather than SDXL, 512px rather than 1024, and three memory switches that
together are the difference between running and an out-of-memory crash.

Nothing here is imported until you construct a Generator, so the rest of the
package stays runnable on a machine with no torch at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bot.bootstrap import SD15, Family
from bot.persona import Persona

IP_ADAPTER_REPO = "h94/IP-Adapter"

# How strongly the anchor pulls. Too high and every picture copies the anchor's
# pose; too low and she stops being recognisably herself between scenes.
DEFAULT_SCALE = 0.65


@dataclass
class Generator:
    """A loaded pipeline, conditioned on the anchor. Construct once, reuse."""

    persona: Persona
    family: Family = SD15
    scale: float = DEFAULT_SCALE
    steps: int = 28
    guidance: float = 7.0
    low_memory: bool = True
    lora: Path | None = None

    def __post_init__(self) -> None:
        import torch
        from diffusers import AutoPipelineForText2Image
        from PIL import Image

        anchor_path = self.persona.visual.anchor_image
        if not anchor_path.exists():
            raise FileNotFoundError(
                f"anchor missing at {anchor_path} -- every picture is conditioned on it"
            )

        self._torch = torch
        self._anchor = Image.open(anchor_path).convert("RGB")
        pipe = AutoPipelineForText2Image.from_pretrained(
            self.family.base, torch_dtype=torch.float16, safety_checker=None,
        )
        pipe.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder=self.family.ip_adapter_subfolder,
            weight_name=self.family.ip_adapter_weight,
        )
        pipe.set_ip_adapter_scale(self.scale)
        if self.lora:
            pipe.load_lora_weights(str(self.lora))

        if self.low_memory:
            # On 6GB these three are what make it run at all: weights move to
            # the GPU a submodule at a time, and attention and VAE work in
            # slices rather than allocating one enormous tensor.
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        else:
            pipe.to("cuda")

        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe

    @property
    def resolution(self) -> int:
        return self.family.resolution

    def render(self, prompt: str, negative: str, seed: int, scale: float | None = None):
        if scale is not None:
            self._pipe.set_ip_adapter_scale(scale)
        try:
            return self._pipe(
                prompt=prompt,
                negative_prompt=negative,
                ip_adapter_image=self._anchor,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                height=self.resolution,
                width=self.resolution,
                generator=self._torch.Generator("cuda").manual_seed(int(seed)),
            ).images[0]
        finally:
            if scale is not None:
                self._pipe.set_ip_adapter_scale(self.scale)

    def generate(self, shot: dict, target: Path) -> None:
        """The callable `bot.bootstrap.run_batch` expects."""
        image = self.render(shot["prompt"], shot["negative"], shot["seed"])
        image.save(target)

    def sweep(self, shot: dict, out_dir: Path, scales=(0.4, 0.55, 0.7, 0.85)) -> list[Path]:
        """One prompt at several anchor strengths, so you can pick by eye.

        Worth the few minutes: this single number decides whether the whole
        batch is usable.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for value in scales:
            path = out_dir / f"sweep-{value}.png"
            self.render(shot["prompt"], shot["negative"], shot["seed"], scale=value).save(path)
            written.append(path)
        return written
