"""Executes the bootstrap orchestration end to end with a stub generator.

The diffusers calls are the only part this cannot cover; everything around
them -- resume, failure tolerance, gating, curation, the report -- runs here.
"""

import json
from pathlib import Path

import pytest

from bot.bootstrap import (
    assemble_training_set,
    gate_images,
    run_batch,
    shot_path,
    write_report,
)
from bot.qc import Judgement


def shots(n):
    return [{"index": i, "prompt": f"prompt {i}", "negative": "neg", "seed": i} for i in range(n)]


def stub_generator(calls=None):
    def generate(shot, target):
        if calls is not None:
            calls.append(shot["index"])
        target.write_bytes(b"fake-image-" + str(shot["index"]).encode())
    return generate


class StubEmbedder:
    """Anchor-like images embed near (1,0); the odd ones drift away."""

    def embed(self, image):
        name = Path(image).name
        if "anchor" in name:
            return [1.0, 0.0]
        n = int("".join(c for c in name if c.isdigit()) or 0)
        return [1.0, 0.0] if n % 3 else [0.0, 1.0]


# -- generation ----------------------------------------------------------


def test_every_shot_is_generated(tmp_path):
    result = run_batch(shots(5), stub_generator(), tmp_path)
    assert len(result.generated) == 5
    assert all(p.exists() for p in result.generated)


def test_files_are_named_by_index(tmp_path):
    run_batch(shots(2), stub_generator(), tmp_path)
    assert (tmp_path / "shot-0000.png").exists()
    assert (tmp_path / "shot-0001.png").exists()


def test_output_directory_is_created(tmp_path):
    nested = tmp_path / "deep" / "out"
    run_batch(shots(1), stub_generator(), nested)
    assert nested.exists()


def test_existing_images_are_skipped(tmp_path):
    """Kaggle sessions time out; a second run must resume, not restart."""
    run_batch(shots(3), stub_generator(), tmp_path)
    calls = []
    second = run_batch(shots(5), stub_generator(calls), tmp_path)
    assert calls == [3, 4], "should only generate what is missing"
    assert len(second.skipped) == 3
    assert len(second.images) == 5


def test_empty_files_are_regenerated(tmp_path):
    """A zero-byte file is a half-written image, not a finished one."""
    shot_path(tmp_path, {"index": 0}).write_bytes(b"")
    calls = []
    run_batch(shots(1), stub_generator(calls), tmp_path)
    assert calls == [0]


def test_one_failure_does_not_abandon_the_batch(tmp_path):
    def flaky(shot, target):
        if shot["index"] == 2:
            raise RuntimeError("cuda oom")
        target.write_bytes(b"ok")

    result = run_batch(shots(5), flaky, tmp_path)
    assert len(result.generated) == 4
    assert result.failed == ((2, "cuda oom"),)
    assert "1 failed" in result.summary()


def test_a_generator_writing_nothing_is_recorded_as_failure(tmp_path):
    result = run_batch(shots(2), lambda shot, target: None, tmp_path)
    assert len(result.failed) == 2
    assert all("wrote no file" in reason for _, reason in result.failed)


def test_progress_callback_reports_position(tmp_path):
    seen = []
    run_batch(shots(3), stub_generator(), tmp_path, on_progress=lambda n, total: seen.append((n, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_summary_mentions_what_happened(tmp_path):
    result = run_batch(shots(2), stub_generator(), tmp_path)
    assert "2 generated" in result.summary()


# -- gating --------------------------------------------------------------


@pytest.fixture
def gated(tmp_path):
    anchor = tmp_path / "anchor.png"
    anchor.write_bytes(b"anchor")
    result = run_batch(shots(9), stub_generator(), tmp_path)
    judgements, vectors = gate_images(result.images, anchor, StubEmbedder(), 0.62)
    return judgements, vectors, tmp_path


def test_off_model_images_are_rejected(gated):
    judgements, _, _ = gated
    assert any(not j.passed for j in judgements)
    assert any(j.passed for j in judgements)


def test_gating_covers_every_image(gated):
    judgements, vectors, _ = gated
    assert len(judgements) == 9 and len(vectors) == 9


# -- curation ------------------------------------------------------------


def test_training_set_contains_only_passing_images(gated):
    judgements, vectors, tmp = gated
    picked = assemble_training_set(judgements, vectors, tmp / "train", keep=10)
    assert picked and all(j.passed for j in picked)


def test_training_directory_holds_the_picked_files(gated):
    judgements, vectors, tmp = gated
    picked = assemble_training_set(judgements, vectors, tmp / "train", keep=2)
    files = sorted(p.name for p in (tmp / "train").iterdir())
    assert files == sorted(j.image.name for j in picked)


def test_training_directory_is_rebuilt_not_appended(gated):
    judgements, vectors, tmp = gated
    train = tmp / "train"
    train.mkdir()
    (train / "stale.png").write_bytes(b"left over from a previous run")
    assemble_training_set(judgements, vectors, train, keep=2)
    assert not (train / "stale.png").exists()


def test_curation_respects_the_keep_limit(gated):
    judgements, vectors, tmp = gated
    assert len(assemble_training_set(judgements, vectors, tmp / "train", keep=1)) == 1


# -- report --------------------------------------------------------------


def test_report_records_scores_and_what_was_trained_on(gated):
    judgements, vectors, tmp = gated
    picked = assemble_training_set(judgements, vectors, tmp / "train", keep=2)
    write_report(judgements, picked, tmp / "qc.json")

    report = json.loads((tmp / "qc.json").read_text())
    assert report["kept"] == len(picked)
    assert len(report["images"]) == len(judgements)
    trained = [r for r in report["images"] if r["trained_on"]]
    assert len(trained) == len(picked)
    scores = [r["score"] for r in report["images"]]
    assert scores == sorted(scores, reverse=True), "report should rank by score"


# -- trainer invocation --------------------------------------------------
# The training run itself needs a GPU, but a mistyped flag costs an hour of
# one. These assert the command before it is ever spent.

from bot.bootstrap import SD15, SDXL, family_for_vram, train_command  # noqa: E402


def flags(cmd):
    return {c.split("=", 1)[0]: c.split("=", 1)[1] for c in cmd if c.startswith("--") and "=" in c}


def test_command_launches_the_maintained_trainer():
    cmd = train_command(Path("train"), Path("out"), instance_token="remyx")
    assert cmd[:2] == ["accelerate", "launch"]
    assert cmd[2].endswith("train_dreambooth_lora.py")


# -- which family fits the card -----------------------------------------
# A 6GB GTX 1060 cannot hold SDXL. Picking wrong costs a multi-gigabyte
# download and an out-of-memory crash after it.


@pytest.mark.parametrize("vram,expected", [
    (4096, "sd15"), (6144, "sd15"), (7000, "sdxl"), (8192, "sdxl"), (24576, "sdxl"),
])
def test_family_follows_vram(vram, expected):
    assert family_for_vram(vram).name == expected


def test_sd15_is_the_default():
    assert train_command(Path("t"), Path("o"), instance_token="x")[2].endswith(
        "train_dreambooth_lora.py")


def test_sd15_trains_at_512_and_sdxl_at_1024():
    assert flags(train_command(Path("t"), Path("o"), instance_token="x"))["--resolution"] == "512"
    assert flags(train_command(Path("t"), Path("o"), instance_token="x",
                               family=SDXL))["--resolution"] == "1024"


def test_only_sdxl_gets_the_replacement_vae():
    """SD 1.5's stock VAE is fine in fp16; SDXL's is not."""
    assert "--pretrained_vae_model_name_or_path" not in flags(
        train_command(Path("t"), Path("o"), instance_token="x"))
    assert flags(train_command(Path("t"), Path("o"), instance_token="x", family=SDXL)
                 )["--pretrained_vae_model_name_or_path"] == SDXL.vae


def test_each_family_names_its_own_base_and_adapter():
    assert flags(train_command(Path("t"), Path("o"), instance_token="x")
                 )["--pretrained_model_name_or_path"] == SD15.base
    assert SD15.ip_adapter_weight.endswith("_sd15.bin")
    assert SD15.ip_adapter_subfolder == "models"
    assert SDXL.ip_adapter_subfolder == "sdxl_models"


def test_command_points_at_the_data_and_output():
    cmd = train_command(Path("train"), Path("out"), instance_token="remyx")
    assert flags(cmd)["--instance_data_dir"] == "train"
    assert flags(cmd)["--output_dir"] == "out"


def test_instance_prompt_carries_the_rare_token():
    cmd = train_command(Path("t"), Path("o"), instance_token="remyx")
    assert "remyx" in flags(cmd)["--instance_prompt"]


def test_blank_token_is_rejected():
    with pytest.raises(ValueError, match="rare word"):
        train_command(Path("t"), Path("o"), instance_token="  ")


def test_memory_flags_are_present_for_a_16gb_gpu():
    cmd = train_command(Path("t"), Path("o"), instance_token="remyx")
    for flag in ("--gradient_checkpointing", "--use_8bit_adam"):
        assert flag in cmd
    assert flags(cmd)["--mixed_precision"] == "fp16"
    assert flags(cmd)["--train_batch_size"] == "1"


def test_tunables_are_overridable():
    cmd = train_command(Path("t"), Path("o"), instance_token="x", steps=800, rank=32,
                        learning_rate=5e-5, resolution=768, seed=7)
    f = flags(cmd)
    assert f["--max_train_steps"] == "800" and f["--rank"] == "32"
    assert f["--resolution"] == "768" and f["--seed"] == "7"
    assert float(f["--learning_rate"]) == 5e-5


def test_no_flag_is_passed_twice():
    cmd = train_command(Path("t"), Path("o"), instance_token="remyx")
    names = [c.split("=", 1)[0] for c in cmd if c.startswith("--")]
    assert len(names) == len(set(names))
