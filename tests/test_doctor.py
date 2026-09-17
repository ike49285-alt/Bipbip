import pytest

from bot.doctor import Finding, Report, render


def report_with(vram=0, gpu="", findings=()):
    r = Report(vram_mb=vram, gpu=gpu)
    r.add(*findings)
    return r


# -- the VRAM decision ---------------------------------------------------
# This is the whole point of the command: how much memory decides which
# image model is viable, and getting it wrong wastes a long download.


def test_no_gpu_means_no_image_plan():
    model, why = report_with().image_plan
    assert model == "none"
    assert "CPU" in why


@pytest.mark.parametrize("vram,expected", [
    (24576, "SDXL + LoRA"),
    (12288, "SDXL + LoRA"),
    (8192, "SDXL + LoRA, memory-saving on"),
    (6144, "SD 1.5 + LoRA at 768px"),
    (4096, "SD 1.5 at 512px"),
])
def test_model_choice_follows_vram(vram, expected):
    assert report_with(vram=vram, gpu="NVIDIA").image_plan[0] == expected


def test_the_reason_quotes_the_actual_memory():
    _, why = report_with(vram=8192, gpu="NVIDIA").image_plan
    assert "8GB" in why


def test_plan_boundaries_do_not_overlap():
    """Every VRAM size gets exactly one plan, with no gap between tiers."""
    plans = {report_with(vram=mb, gpu="x").image_plan[0] for mb in range(1024, 25600, 512)}
    assert "none" not in plans
    assert len(plans) == 4


# -- findings ------------------------------------------------------------


def test_a_failing_finding_shows_its_fix():
    text = str(Finding("torch", False, "not installed", "pip install torch"))
    assert "[no ]" in text and "pip install torch" in text


def test_a_passing_finding_hides_the_fix():
    text = str(Finding("torch", True, "installed", "pip install torch"))
    assert "[yes]" in text and "pip install" not in text


# -- the rendered report -------------------------------------------------


def test_report_names_ollama_as_the_writer_when_running():
    r = report_with(vram=8192, gpu="NVIDIA", findings=[
        Finding("ollama", True, "running, 1 model(s): llama3.1"),
        Finding("ANTHROPIC_API_KEY", False, "not set"),
    ])
    assert "Writing: ollama, locally, free" in render(r)


def test_report_falls_back_to_the_api_when_ollama_is_absent():
    r = report_with(vram=8192, gpu="NVIDIA", findings=[
        Finding("ollama", False, "not running"),
        Finding("ANTHROPIC_API_KEY", True, "set"),
    ])
    assert "Writing: Anthropic API" in render(r)


def test_report_says_when_nothing_can_write():
    r = report_with(vram=8192, gpu="NVIDIA", findings=[
        Finding("ollama", False, "not running"),
        Finding("ANTHROPIC_API_KEY", False, "not set"),
    ])
    assert "nothing configured yet" in render(r)


def test_report_collects_the_blocking_fixes():
    r = report_with(findings=[
        Finding("GPU", False, "missing", "install the driver"),
        Finding("torch", False, "missing", "pip install torch"),
        Finding("disk", False, "full", "free some space"),
    ])
    out = render(r)
    assert "Before images will work:" in out
    assert "install the driver" in out and "pip install torch" in out
    # disk is not an image blocker; it is reported but not listed as one
    assert out.count("free some space") == 1


def test_report_omits_the_blocker_list_when_ready():
    r = report_with(vram=12288, gpu="NVIDIA", findings=[
        Finding("GPU", True, "ok"), Finding("torch", True, "ok"),
        Finding("diffusers", True, "ok"), Finding("ollama", True, "running"),
    ])
    assert "Before images will work" not in render(r)
