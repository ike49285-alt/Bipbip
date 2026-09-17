"""What can this machine actually run?

Answers the two questions that decide the whole local setup: which image model
fits in this GPU's memory, and what is writing the words. Deliberately depends
on nothing -- it runs before anything is installed, which is the point.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Finding:
    name: str
    ok: bool
    detail: str
    fix: str = ""

    def __str__(self) -> str:
        mark = "yes" if self.ok else "no "
        line = f"  [{mark}] {self.name}: {self.detail}"
        return line + (f"\n         -> {self.fix}" if self.fix and not self.ok else "")


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    vram_mb: int = 0
    gpu: str = ""

    def add(self, *findings: Finding) -> None:
        self.findings.extend(findings)

    @property
    def image_plan(self) -> tuple[str, str]:
        """(model, why) -- picked from VRAM, which is the binding constraint."""
        if not self.gpu:
            return ("none", "no NVIDIA GPU found; images would fall back to CPU at minutes each")
        if self.vram_mb >= 11000:
            return ("SDXL + LoRA", f"{self.vram_mb // 1024}GB is comfortable for SDXL at 1024px")
        if self.vram_mb >= 7000:
            return ("SDXL + LoRA, memory-saving on",
                    f"{self.vram_mb // 1024}GB fits SDXL with attention slicing and VAE tiling")
        if self.vram_mb >= 5000:
            return ("SD 1.5 + LoRA at 768px",
                    f"{self.vram_mb // 1024}GB is tight for SDXL; SD 1.5 trains and runs comfortably")
        return ("SD 1.5 at 512px",
                f"{self.vram_mb // 1024}GB is small; keep resolution low and batch size 1")


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def check_gpu(report: Report) -> None:
    if not shutil.which("nvidia-smi"):
        report.add(Finding(
            "GPU", False, "nvidia-smi not found",
            "sudo ubuntu-drivers autoinstall && reboot  (then re-run)",
        ))
        return
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if not out:
        report.add(Finding("GPU", False, "nvidia-smi present but returned nothing",
                           "check the driver with: nvidia-smi"))
        return
    name, _, mem = out.splitlines()[0].partition(",")
    report.gpu = name.strip()
    try:
        report.vram_mb = int(mem.strip())
    except ValueError:
        report.vram_mb = 0
    report.add(Finding("GPU", True, f"{report.gpu}, {report.vram_mb // 1024}GB VRAM"))


def check_python_stack(report: Report) -> None:
    import importlib.util as iu

    torch_spec = iu.find_spec("torch")
    if torch_spec is None:
        report.add(Finding("torch", False, "not installed",
                           "pip install torch --index-url https://download.pytorch.org/whl/cu121"))
    else:
        cuda = _run(["python3", "-c", "import torch;print(torch.cuda.is_available())"])
        ok = cuda == "True"
        report.add(Finding("torch", ok,
                           "installed, CUDA available" if ok else "installed but CUDA not available",
                           "" if ok else "the CPU build is installed; reinstall the cu121 wheel"))

    for module, install in (("diffusers", "pip install diffusers transformers accelerate safetensors peft"),
                            ("PIL", "pip install pillow")):
        present = iu.find_spec(module) is not None
        report.add(Finding(module, present, "installed" if present else "not installed",
                           "" if present else install))


def check_writer(report: Report) -> None:
    """Who writes the words: a local Ollama, or the Anthropic API."""
    ollama = None
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as r:
            ollama = r.read().decode()
    except (urllib.error.URLError, OSError, TimeoutError):
        pass

    if ollama is not None:
        import json

        try:
            names = [m["name"] for m in json.loads(ollama).get("models", [])]
        except (ValueError, KeyError, TypeError):
            names = []
        report.add(Finding("ollama", True,
                           f"running, {len(names)} model(s): {', '.join(names[:4]) or 'none pulled'}",
                           "" if names else "pull one: ollama pull llama3.1"))
    else:
        report.add(Finding("ollama", False, "not running on :11434",
                           "optional, and free: curl -fsSL https://ollama.com/install.sh | sh"))

    key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    report.add(Finding("ANTHROPIC_API_KEY", key,
                       "set" if key else "not set",
                       "" if key else "optional if you use ollama: export ANTHROPIC_API_KEY=..."))


def check_disk(report: Report) -> None:
    free_gb = shutil.disk_usage(".").free // (1024 ** 3)
    enough = free_gb >= 20
    report.add(Finding("disk", enough, f"{free_gb}GB free",
                       "" if enough else "model weights need ~15GB; free some space first"))


def diagnose() -> Report:
    report = Report()
    check_gpu(report)
    check_python_stack(report)
    check_writer(report)
    check_disk(report)
    return report


def render(report: Report) -> str:
    model, why = report.image_plan
    lines = ["This machine:", ""]
    lines += [str(f) for f in report.findings]
    lines += ["", f"Images:  {model}", f"         {why}"]

    writer = next((f for f in report.findings if f.name == "ollama" and f.ok), None)
    key = next((f for f in report.findings if f.name == "ANTHROPIC_API_KEY" and f.ok), None)
    if writer:
        lines.append("Writing: ollama, locally, free")
    elif key:
        lines.append("Writing: Anthropic API")
    else:
        lines += ["Writing: nothing configured yet",
                  "         install ollama (free, local) or set ANTHROPIC_API_KEY"]

    blocked = [f for f in report.findings if not f.ok and f.name in {"GPU", "torch", "diffusers"}]
    if blocked:
        lines += ["", "Before images will work:"]
        lines += [f"  {f.fix}" for f in blocked if f.fix]
    return "\n".join(lines)
