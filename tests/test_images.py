import base64
import json

import pytest

from thirsttrap import images
from thirsttrap.images import (
    BOUNDS,
    ImageError,
    OpenAIImageBackend,
    PersonaRejected,
    PollinationsBackend,
    TogetherBackend,
    build,
    build_prompt,
    detect,
    validate_look,
)


class TestBounds:
    def test_bounds_are_always_appended(self):
        assert build_prompt("a woman on a balcony").endswith(BOUNDS)

    def test_the_persona_comes_before_the_scene(self):
        prompt = build_prompt("dark curly hair", "on a fire escape")
        assert prompt.index("dark curly hair") < prompt.index("on a fire escape")

    def test_an_empty_scene_is_dropped(self):
        assert ", , " not in build_prompt("dark curly hair", "")


class TestValidation:
    @pytest.mark.parametrize(
        "text", ["a teen girl", "schoolgirl outfit", "young girl, 17", "underage"]
    )
    def test_anything_suggesting_a_minor_is_refused(self, text):
        # Appending "adult" to the prompt only contradicts this; refusing does not.
        with pytest.raises(PersonaRejected):
            validate_look(text)

    @pytest.mark.parametrize("text", ["nude on a beach", "topless", "explicit photo", "nsfw"])
    def test_explicit_requests_are_refused(self, text):
        with pytest.raises(PersonaRejected):
            validate_look(text)

    def test_the_refusal_says_which_word_tripped_it(self):
        with pytest.raises(PersonaRejected, match="teen"):
            validate_look("a teen in a sundress")

    @pytest.mark.parametrize(
        "text",
        ["dark curly hair, freckles", "a woman in a linen shirt", "golden hour, film grain"],
    )
    def test_ordinary_descriptions_pass(self, text):
        validate_look(text)

    def test_build_prompt_validates_before_returning(self):
        with pytest.raises(PersonaRejected):
            build_prompt("dark hair", "nude in the sea")

    def test_case_and_spacing_do_not_evade_it(self):
        with pytest.raises(PersonaRejected):
            validate_look("A   TEEN   girl")


class TestBackendSelection:
    @pytest.mark.parametrize(
        "name,cls",
        [("pollinations", PollinationsBackend), ("together", TogetherBackend),
         ("openai-compat", OpenAIImageBackend)],
    )
    def test_names_map_to_backends(self, name, cls):
        assert isinstance(build(name), cls)

    def test_an_unknown_name_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="pollinations"):
            build("dall-e-2000")

    def test_a_model_override_is_applied(self):
        assert build("together", model="my-model").model == "my-model"

    def test_the_keyless_backend_is_the_default(self, monkeypatch):
        monkeypatch.delenv(images.ENV_BACKEND, raising=False)
        monkeypatch.delenv(images.ENV_KEY, raising=False)
        assert isinstance(detect(), PollinationsBackend)

    def test_a_key_in_the_environment_selects_together(self, monkeypatch):
        monkeypatch.delenv(images.ENV_BACKEND, raising=False)
        monkeypatch.setenv(images.ENV_KEY, "sk-test")
        assert isinstance(detect(), TogetherBackend)

    def test_keyed_backends_report_unavailable_without_one(self):
        assert not TogetherBackend().available()
        assert TogetherBackend(key="k").available()
        assert PollinationsBackend().available()


class TestResponseDecoding:
    def test_base64_payloads_decode(self):
        raw = b"\xff\xd8fake jpeg"
        data = {"data": [{"b64_json": base64.b64encode(raw).decode()}]}
        assert images._decode_openai_shape(data, "test") == raw

    def test_an_empty_response_raises(self):
        with pytest.raises(ImageError, match="no image"):
            images._decode_openai_shape({"data": []}, "test")

    def test_a_response_with_neither_field_raises(self):
        with pytest.raises(ImageError, match="neither"):
            images._decode_openai_shape({"data": [{"revised_prompt": "x"}]}, "test")

    def test_unreadable_base64_raises(self):
        with pytest.raises(ImageError, match="base64"):
            images._decode_openai_shape({"data": [{"b64_json": "!!!not base64!!!"}]}, "test")
