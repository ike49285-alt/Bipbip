import json

import pytest

from bot import run, x
from bot.persona import Persona
from bot.run import Post, choose, load_history, publish, record, save_history

PERSONA = Persona(
    name="Mira", look="dark curly hair",
    scenes=["on a fire escape", "in a laundrette", "on a balcony"],
    topics=["quitting", "the walk home", "leaving first"],
)


class TestHistory:
    def test_a_missing_log_is_empty_not_fatal(self, tmp_path):
        assert load_history(tmp_path / "absent.json") == []

    def test_a_damaged_log_is_empty_not_fatal(self, tmp_path):
        path = tmp_path / "posted.json"
        path.write_text("{not json")
        assert load_history(path) == []

    def test_a_json_object_is_rejected_without_crashing(self, tmp_path):
        path = tmp_path / "posted.json"
        path.write_text('{"a": 1}')
        assert load_history(path) == []

    def test_it_round_trips(self, tmp_path):
        path = tmp_path / "posted.json"
        save_history([{"caption": "one"}], path)
        assert load_history(path) == [{"caption": "one"}]

    def test_it_is_capped_so_it_cannot_grow_forever(self, tmp_path):
        path = tmp_path / "posted.json"
        save_history([{"caption": str(i)} for i in range(run.KEEP + 20)], path)
        rows = load_history(path)
        assert len(rows) == run.KEEP
        assert rows[-1]["caption"] == str(run.KEEP + 19)


class TestChoosing:
    def test_the_same_seed_gives_the_same_pair(self):
        assert choose(PERSONA, [], seed=5) == choose(PERSONA, [], seed=5)

    def test_it_picks_from_the_persona(self):
        scene, topic = choose(PERSONA, [], seed=3)
        assert scene in PERSONA.scenes and topic in PERSONA.topics

    def test_it_avoids_a_recently_used_scene(self):
        history = [{"scene": "on a fire escape", "topic": "quitting"}]
        scenes = {choose(PERSONA, history, seed=s)[0] for s in range(12)}
        assert "on a fire escape" not in scenes

    def test_it_still_picks_when_everything_is_recent(self):
        # Exhausting the persona must not deadlock or return nothing.
        history = [{"scene": s, "topic": t} for s in PERSONA.scenes for t in PERSONA.topics]
        scene, topic = choose(PERSONA, history, seed=1)
        assert scene in PERSONA.scenes and topic in PERSONA.topics


class TestPublishing:
    def test_missing_credentials_name_the_secrets(self):
        post = Post(scene="s", topic="t", caption="c", alt="a", prompt="p", image=b"\xff\xd8x")
        with pytest.raises(x.XError, match="X_API_KEY"):
            publish(post, creds=x.Credentials())

    def test_a_post_with_no_image_is_refused(self):
        post = Post(scene="s", topic="t", caption="c", alt="a", prompt="p")
        with pytest.raises(x.XError, match="no image"):
            publish(post, creds=x.Credentials("a", "b", "c", "d"))


class TestRecording:
    def test_it_appends_what_a_later_run_needs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run, "HISTORY", tmp_path / "posted.json")
        post = Post(scene="on a balcony", topic="quitting", caption="You already know.",
                    alt="a", prompt="p")
        record(post, "1234567890", history=[])

        rows = load_history(tmp_path / "posted.json")
        assert len(rows) == 1
        assert rows[0]["id"] == "1234567890"
        assert rows[0]["scene"] == "on a balcony"
        assert rows[0]["caption"] == "You already know."
        assert rows[0]["at"].endswith("+00:00")
