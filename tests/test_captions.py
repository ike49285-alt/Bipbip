import pytest

from bot.captions import CaptionError, parse, pick, preference, similar, usable


class TestParsing:
    def test_a_json_array(self):
        assert parse('["one", "two"]') == ["one", "two"]

    def test_a_fenced_array(self):
        assert parse('```json\n["one"]\n```') == ["one"]

    def test_a_numbered_list_when_the_model_drifts(self):
        assert parse("Here are 2:\n1. one\n2) two") == ["one", "two"]

    def test_bullets(self):
        assert parse("- one\n* two") == ["one", "two"]

    def test_wrapping_quotes_are_stripped(self):
        assert parse('"one"\n“two”') == ["one", "two"]

    def test_empty_input_yields_nothing(self):
        assert parse("") == []


class TestUsable:
    @pytest.mark.parametrize(
        "caption,ok",
        [
            ("You already know the answer. You're stalling.", True),
            ("short", False),                                   # too short to say anything
            ("x" * 300, False),                                  # over the tweet limit
            ("check this out #viral", False),
            ("dm me @someone about it here", False),
            ("read more at https://example.com now", False),
            ("so good \U0001F525 really the best", False),
            ("wow!! amazing!! incredible!!", False),
        ],
    )
    def test_hard_rules(self, caption, ok):
        assert usable(caption) is ok


class TestPreference:
    def test_short_is_preferred(self):
        assert preference("You already know. You're stalling.") > preference("You " + "word " * 40)

    def test_second_person_is_preferred(self):
        assert preference("You already know it.") > preference("Someone already knows it.")

    def test_a_short_closing_line_is_preferred(self):
        assert preference("Three years of that job. Gone.") > preference(
            "Three years of that job and then it ended in a fairly long sentence."
        )

    def test_tired_phrases_cost(self):
        assert preference("let that sink in, you know") < preference("you know exactly what it is")


class TestPicking:
    def test_it_returns_the_best_usable_one(self):
        chosen = pick([
            "#spam #spam",
            "You already know the answer. You're stalling.",
            "The weather this week has been fairly agreeable on balance, one supposes.",
        ])
        assert chosen == "You already know the answer. You're stalling."

    def test_nothing_usable_raises(self):
        with pytest.raises(CaptionError, match="usable"):
            pick(["#spam", "short"])

    def test_it_avoids_repeating_a_recent_post(self):
        old = "You already know the answer. You're stalling."
        chosen = pick([old, "Quit at four. Home by five. Never felt lighter."], avoid=[old])
        assert chosen != old

    def test_everything_being_a_rerun_raises(self):
        old = "You already know the answer. You're stalling."
        with pytest.raises(CaptionError, match="new"):
            pick([old], avoid=[old])

    def test_similarity_is_symmetric_and_bounded(self):
        a, b = "cold coffee and rain", "rain and warm coffee"
        assert similar(a, b) == similar(b, a)
        assert similar(a, a) == 1.0 and similar(a, "nothing alike") >= 0.0


class TestModelListing:
    def test_junk_entries_are_filtered_out(self):
        from bot.captions import NOT_CHAT

        for junk in ["whisper-large-v3", "llama-guard-4-12b", "text-embedding-3-small",
                     "playai-tts", "rerank-v1", "omni-moderation-latest"]:
            assert NOT_CHAT.search(junk), junk

    def test_chat_models_survive_the_filter(self):
        from bot.captions import NOT_CHAT

        for good in ["llama-3.3-70b-versatile", "gemma2-9b-it", "gpt-4o", "qwen2.5-72b"]:
            assert not NOT_CHAT.search(good), good

    def test_the_listing_url_is_derived_from_the_chat_url(self, monkeypatch):
        import re

        from bot import captions

        monkeypatch.setenv(captions.ENV_URL, "https://api.groq.com/openai/v1/chat/completions")
        url, _, _ = captions._config_url_key()
        assert re.sub(r"/chat/completions/?$", "/models", url) == \
            "https://api.groq.com/openai/v1/models"

    def test_listing_does_not_require_a_model_to_be_set(self, monkeypatch):
        # The whole point is to find out what to set CAPTION_MODEL to.
        from bot import captions

        monkeypatch.delenv(captions.ENV_MODEL, raising=False)
        url, _, model = captions._config_url_key()
        assert url and model == ""
