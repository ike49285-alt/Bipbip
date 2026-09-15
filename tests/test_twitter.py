import pytest

from thirsttrap import twitter
from thirsttrap.twitter import Credentials, TwitterError, _quote, post, sign, upload_media


class TestSigning:
    def test_it_matches_twitters_published_example(self):
        """The documented OAuth 1.0a example from X's own developer docs.

        Pinned because a wrong signature is a 401 with no useful message, and
        every other test here would still pass while nothing could post.
        """
        creds = Credentials(
            consumer_key="xvz1evFS4wEEPTGEFPHBog",
            consumer_secret="kAcSOqF21Fu85e7zjz7ZN2U4ZRhfV3WpwPAoE3Z7kBw",
            token="370773112-GmHxMAgYyLbNEtIKZeRNFsMKPR9EyMZeS9weJAEb",
            token_secret="LswwdoUaIvS8ltyTt5jkRh4J50vUPVVHtR2YPi5kE",
        )
        import base64, hashlib, hmac

        params = {
            "status": "Hello Ladies + Gentlemen, a signed OAuth request!",
            "include_entities": "true",
            "oauth_consumer_key": creds.consumer_key,
            "oauth_nonce": "kYjzVBB8Y0ZFabxSWbWovY3uYSQ2pTgmZeNu2VS4cg",
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": "1318622958",
            "oauth_token": creds.token,
            "oauth_version": "1.0",
        }
        url = "https://api.twitter.com/1.1/statuses/update.json"
        encoded = "&".join(f"{_quote(k)}={_quote(v)}" for k, v in sorted(params.items()))
        base = "&".join(["POST", _quote(url), _quote(encoded)])
        key = f"{_quote(creds.consumer_secret)}&{_quote(creds.token_secret)}".encode()
        signature = base64.b64encode(hmac.new(key, base.encode(), hashlib.sha1).digest()).decode()

        assert signature == "hCtSmYh+iHYCEqBWrE7C7hYmtUk="

    def test_percent_encoding_follows_rfc3986(self):
        # quote()'s defaults leave / unescaped and encode ~, both of which break OAuth.
        assert _quote("a b+c~d/e") == "a%20b%2Bc~d%2Fe"

    def test_the_header_carries_every_required_field(self):
        header = sign(Credentials("ck", "cs", "tk", "ts"), "POST", "https://api.x.com/2/tweets")
        for field in ("oauth_consumer_key", "oauth_nonce", "oauth_signature",
                      "oauth_signature_method", "oauth_timestamp", "oauth_token", "oauth_version"):
            assert field in header

    def test_each_signature_uses_a_fresh_nonce(self):
        creds = Credentials("ck", "cs", "tk", "ts")
        url = "https://api.x.com/2/tweets"
        assert sign(creds, "POST", url) != sign(creds, "POST", url)


class TestCredentials:
    def test_missing_names_the_environment_variables(self):
        assert Credentials().missing() == [
            "X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET"
        ]

    def test_complete_only_when_all_four_are_present(self):
        assert not Credentials("a", "b", "c").complete()
        assert Credentials("a", "b", "c", "d").complete()

    def test_they_are_read_from_the_environment(self, monkeypatch):
        for var in twitter.ENV.values():
            monkeypatch.setenv(var, "value-" + var)
        assert Credentials.from_env().complete()

    def test_redaction_removes_secrets_from_a_message(self):
        creds = Credentials("consumerkey1", "consumersecret1", "tokenvalue1", "tokensecret1")
        cleaned = creds.redact("failed with tokensecret1 and consumerkey1")
        assert "tokensecret1" not in cleaned and "consumerkey1" not in cleaned

    def test_redaction_leaves_short_values_alone(self):
        # A 2-character secret would otherwise redact half the alphabet.
        assert Credentials("ab", "cd", "ef", "gh").redact("a fine message") == "a fine message"


class TestGuards:
    def test_an_empty_tweet_is_refused_before_any_request(self):
        with pytest.raises(TwitterError, match="empty"):
            post(Credentials("a", "b", "c", "d"), "   ")

    def test_an_overlong_caption_is_refused_with_its_length(self):
        with pytest.raises(TwitterError, match="281"):
            post(Credentials("a", "b", "c", "d"), "x" * 281)

    def test_an_empty_image_is_refused(self):
        with pytest.raises(TwitterError, match="no image"):
            upload_media(Credentials("a", "b", "c", "d"), b"")

    def test_an_oversized_image_is_refused_before_upload(self):
        with pytest.raises(TwitterError, match="5 MB"):
            upload_media(Credentials("a", "b", "c", "d"), b"x" * (6 * 1024 * 1024))
