import json

import pytest

from thirsttrap import twitter
from thirsttrap.bot import ALT_PREFIX, Draft, Persona, build, choose, publish, write_caption
from thirsttrap.images import ImageError, PersonaRejected
from thirsttrap.llm import GrammarBackend

GOOD = {
    "name": "Mira",
    "look": "dark curly hair, freckles, film grain",
    "voice": "flirt",
    "scenes": ["on a fire escape", "in a laundrette at midnight"],
    "topics": ["quitting a job", "the walk home at 3am"],
}


def write(tmp_path, **overrides):
    data = {**GOOD, **overrides}
    path = tmp_path / "persona.json"
    path.write_text(json.dumps(data))
    return path


class FakeImages:
    def __init__(self, payload=b"\xff\xd8jpeg-bytes", boom=None):
        self.payload, self.boom, self.calls = payload, boom, []

    def render(self, prompt, seed=None):
        self.calls.append({"prompt": prompt, "seed": seed})
        if self.boom:
            raise self.boom
        return self.payload

    def describe(self):
        return "fake images"


class TestPersonaFile:
    def test_a_good_file_loads(self, tmp_path):
        persona = Persona.load(write(tmp_path))
        assert persona.name == "Mira" and len(persona.scenes) == 2

    @pytest.mark.parametrize("field", ["look", "scenes", "topics"])
    def test_a_missing_field_says_which(self, tmp_path, field):
        with pytest.raises(ValueError, match=field):
            Persona.load(write(tmp_path, **{field: "" if field == "look" else []}))

    def test_an_unknown_voice_fails_at_load_not_mid_run(self, tmp_path):
        with pytest.raises(KeyError):
            Persona.load(write(tmp_path, voice="smoulder"))

    def test_a_persona_asking_for_a_minor_is_refused_at_load(self, tmp_path):
        # Before any API call is made, so a bad file costs nothing.
        with pytest.raises(PersonaRejected):
            Persona.load(write(tmp_path, look="a teen girl with freckles"))

    def test_a_persona_asking_for_explicit_content_is_refused_at_load(self, tmp_path):
        with pytest.raises(PersonaRejected):
            Persona.load(write(tmp_path, look="nude, dark curly hair"))

    def test_a_json_array_is_rejected(self, tmp_path):
        path = tmp_path / "persona.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="JSON object"):
            Persona.load(path)

    def test_the_shipped_persona_file_is_valid(self):
        persona = Persona.load("persona.json")
        assert persona.look and persona.scenes and persona.topics


class TestChoosing:
    def test_the_same_seed_picks_the_same_pair(self, tmp_path):
        persona = Persona.load(write(tmp_path))
        assert choose(persona, seed=7)[:2] == choose(persona, seed=7)[:2]

    def test_it_picks_from_the_persona(self, tmp_path):
        persona = Persona.load(write(tmp_path))
        scene, topic, _ = choose(persona, seed=3)
        assert scene in persona.scenes and topic in persona.topics


class TestBuilding:
    def test_a_draft_has_everything_a_post_needs(self, tmp_path):
        persona = Persona.load(write(tmp_path))
        fake = FakeImages()
        draft = build(persona, image_backend=fake, llm_backend=GrammarBackend(), seed=11)

        assert draft.caption and draft.image == b"\xff\xd8jpeg-bytes"
        assert draft.scene in persona.scenes
        assert draft.prompt.startswith(persona.look)

    def test_the_caption_fits_in_a_tweet(self, tmp_path):
        persona = Persona.load(write(tmp_path))
        for seed in range(6):
            draft = build(persona, image_backend=FakeImages(),
                          llm_backend=GrammarBackend(), seed=seed, render=False)
            assert len(draft.caption) <= twitter.MAX_TWEET

    def test_alt_text_always_says_the_image_is_ai_generated(self, tmp_path):
        # Costs none of the 280 characters and tells anyone who checks.
        persona = Persona.load(write(tmp_path))
        draft = build(persona, llm_backend=GrammarBackend(), seed=2, render=False)
        assert draft.alt_text.startswith(ALT_PREFIX)
        assert persona.name in draft.alt_text

    def test_render_false_skips_the_image_call_entirely(self, tmp_path):
        persona = Persona.load(write(tmp_path))
        fake = FakeImages()
        draft = build(persona, image_backend=fake, llm_backend=GrammarBackend(),
                      seed=1, render=False)
        assert draft.image is None and fake.calls == []

    def test_the_bounds_reach_the_image_backend(self, tmp_path):
        from thirsttrap.images import BOUNDS

        persona = Persona.load(write(tmp_path))
        fake = FakeImages()
        build(persona, image_backend=fake, llm_backend=GrammarBackend(), seed=1)
        assert fake.calls[0]["prompt"].endswith(BOUNDS)

    def test_an_image_failure_surfaces(self, tmp_path):
        persona = Persona.load(write(tmp_path))
        with pytest.raises(ImageError):
            build(persona, image_backend=FakeImages(boom=ImageError("provider down")),
                  llm_backend=GrammarBackend(), seed=1)

    def test_captions_fall_back_to_the_grammar_when_the_model_is_dead(self, tmp_path):
        from thirsttrap.llm import Backend, BackendError

        class Dead(Backend):
            def available(self): return True
            def chat(self, system, messages, json_mode=False): raise BackendError("refused")
            def describe(self): return "dead"

        persona = Persona.load(write(tmp_path))
        ranked = write_caption(persona, "quitting a job", backend=Dead(), seed=4)
        assert ranked and ranked[0].text


class TestPublishing:
    def test_missing_credentials_name_the_secrets_to_set(self):
        draft = Draft(caption="hi", scene="s", topic="t", prompt="p", alt_text="a",
                      image=b"\xff\xd8x")
        with pytest.raises(twitter.TwitterError, match="X_API_KEY"):
            publish(draft, creds=twitter.Credentials())

    def test_a_draft_with_no_image_is_refused(self):
        draft = Draft(caption="hi", scene="s", topic="t", prompt="p", alt_text="a")
        with pytest.raises(twitter.TwitterError, match="no image"):
            publish(draft, creds=twitter.Credentials("a", "b", "c", "d"))
