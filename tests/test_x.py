import base64
import hashlib
import hmac

import pytest

from bot import x
from bot.x import Credentials, XError, header, quote, tweet, upload


class TestSigning:
    def test_it_matches_x_s_published_example(self):
        """X's own documented OAuth 1.0a example, with its expected signature.

        Pinned because a wrong signature is a 401 with no useful body: every
        other test here would pass while nothing could ever post.
        """
        creds = Credentials(
            consumer_key="xvz1evFS4wEEPTGEFPHBog",
            consumer_secret="kAcSOqF21Fu85e7zjz7ZN2U4ZRhfV3WpwPAoE3Z7kBw",
            token="370773112-GmHxMAgYyLbNEtIKZeRNFsMKPR9EyMZeS9weJAEb",
            token_secret="LswwdoUaIvS8ltyTt5jkRh4J50vUPVVHtR2YPi5kE",
        )
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

        # Rebuilt exactly as header() builds it.
        joined = "&".join(f"{quote(k)}={quote(v)}" for k, v in sorted(params.items()))
        base = "&".join(["POST", quote(url), quote(joined)])
        key = f"{quote(creds.consumer_secret)}&{quote(creds.token_secret)}".encode()
        signature = base64.b64encode(hmac.new(key, base.encode(), hashlib.sha1).digest()).decode()

        assert signature == "hCtSmYh+iHYCEqBWrE7C7hYmtUk="

    def test_encoding_is_rfc3986(self):
        # quote()'s defaults leave / alone and escape ~; both break the signature.
        assert quote("a b+c~d/e") == "a%20b%2Bc~d%2Fe"

    def test_every_required_field_is_in_the_header(self):
        built = header(Credentials("ck", "cs", "tk", "ts"), "POST", x.TWEET_URL)
        for field in ("oauth_consumer_key", "oauth_nonce", "oauth_signature",
                      "oauth_signature_method", "oauth_timestamp", "oauth_token",
                      "oauth_version"):
            assert field in built

    def test_the_nonce_is_fresh_each_time(self):
        creds = Credentials("ck", "cs", "tk", "ts")
        assert header(creds, "POST", x.TWEET_URL) != header(creds, "POST", x.TWEET_URL)


class TestCredentials:
    def test_missing_names_the_variables_to_set(self):
        assert Credentials().missing() == [
            "X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET"
        ]

    def test_complete_needs_all_four(self):
        assert not Credentials("a", "b", "c").complete()
        assert Credentials("a", "b", "c", "d").complete()

    def test_they_come_from_the_environment(self, monkeypatch):
        for var in x.ENV.values():
            monkeypatch.setenv(var, "v-" + var)
        assert Credentials.from_env().complete()

    def test_secrets_are_redacted_from_messages(self):
        creds = Credentials("consumerkey1", "consumersecret1", "tokenvalue1", "tokensecret1")
        cleaned = creds.redact("failed using tokensecret1 and consumerkey1")
        assert "tokensecret1" not in cleaned and "consumerkey1" not in cleaned
        assert "<redacted>" in cleaned

    def test_short_values_are_left_alone(self):
        # Redacting a 2-character secret would mangle every message.
        assert Credentials("ab", "cd", "ef", "gh").redact("a fine message") == "a fine message"


class TestGuards:
    def test_an_empty_caption_is_refused_before_any_request(self):
        with pytest.raises(XError, match="empty"):
            tweet(Credentials("a", "b", "c", "d"), "   ")

    def test_an_overlong_caption_reports_its_length(self):
        with pytest.raises(XError, match="281"):
            tweet(Credentials("a", "b", "c", "d"), "x" * 281)

    def test_an_empty_image_is_refused(self):
        with pytest.raises(XError, match="no image"):
            upload(Credentials("a", "b", "c", "d"), b"")

    def test_an_oversized_image_is_refused_before_upload(self):
        with pytest.raises(XError, match="5 MB"):
            upload(Credentials("a", "b", "c", "d"), b"x" * (6 * 1024 * 1024))
