import json

import pytest

from bot.persona import BOUNDS, Persona, Refused, check

GOOD = {
    "name": "Mira",
    "look": "dark curly hair, freckles, film grain",
    "scenes": ["on a fire escape", "in a laundrette at midnight"],
    "topics": ["quitting a job", "the walk home at 3am"],
}


def write(tmp_path, **overrides):
    path = tmp_path / "persona.json"
    path.write_text(json.dumps({**GOOD, **overrides}))
    return path


class TestLoading:
    def test_a_good_file_loads(self, tmp_path):
        persona = Persona.load(write(tmp_path))
        assert persona.name == "Mira" and len(persona.scenes) == 2

    @pytest.mark.parametrize("field", ["name", "look", "scenes", "topics"])
    def test_a_missing_field_names_itself(self, tmp_path, field):
        blank = "" if field in {"name", "look"} else []
        with pytest.raises(ValueError, match=field):
            Persona.load(write(tmp_path, **{field: blank}))

    def test_a_json_array_is_rejected(self, tmp_path):
        path = tmp_path / "persona.json"
        path.write_text("[1, 2]")
        with pytest.raises(ValueError, match="JSON object"):
            Persona.load(path)

    def test_the_shipped_file_is_valid(self):
        persona = Persona.load("persona.json")
        assert persona.scenes and persona.topics


class TestLimits:
    @pytest.mark.parametrize("text", ["a teen girl", "schoolgirl", "young girl", "underage"])
    def test_anything_suggesting_a_minor_is_refused(self, text):
        with pytest.raises(Refused, match="adults"):
            check(text)

    @pytest.mark.parametrize("text", ["nude", "topless", "explicit", "nsfw shot"])
    def test_explicit_requests_are_refused(self, text):
        with pytest.raises(Refused, match="suggestive"):
            check(text)

    def test_the_refusal_names_the_word(self):
        with pytest.raises(Refused, match="teen"):
            check("a teen in a sundress")

    def test_case_and_spacing_do_not_evade_it(self):
        with pytest.raises(Refused):
            check("A   TEEN   girl")

    @pytest.mark.parametrize("text", ["dark curly hair", "a linen shirt, golden hour"])
    def test_ordinary_descriptions_pass(self, text):
        check(text)

    def test_a_bad_look_is_refused_when_the_file_loads(self, tmp_path):
        # Before any generation is billed.
        with pytest.raises(Refused):
            Persona.load(write(tmp_path, look="a teen with freckles"))

    def test_a_bad_scene_is_refused_too(self, tmp_path):
        with pytest.raises(Refused):
            Persona.load(write(tmp_path, scenes=["nude on a beach"]))


class TestPrompt:
    def test_the_bounds_always_end_the_prompt(self, tmp_path):
        persona = Persona.load(write(tmp_path))
        assert persona.prompt("on a balcony").endswith(BOUNDS)

    def test_look_comes_before_scene(self, tmp_path):
        persona = Persona.load(write(tmp_path))
        prompt = persona.prompt("on a balcony")
        assert prompt.index("curly hair") < prompt.index("balcony")

    def test_alt_text_always_declares_the_image(self, tmp_path):
        persona = Persona.load(write(tmp_path))
        alt = persona.alt_text("on a balcony")
        assert alt.startswith("AI-generated image.") and "Mira" in alt
