# bot

Posts one image and one caption a day to X, as a disclosed AI persona. Runs on
GitHub Actions, so it needs no computer of its own. No dependencies — Python
standard library only.

```bash
python -m bot                 # dry run: builds a post, publishes nothing
python -m bot --out p.jpg     # ...and keep the image to look at
python -m bot --show          # ...and show every caption it considered
python -m bot --post          # publish
```

## The persona

`persona.json` — a name, a `look`, a `voice`, and lists of `scenes` and
`topics`. Each run picks a scene and a topic, preferring ones it hasn't used
recently, writes captions about the topic, and generates an image of the look
in the scene.

```json
{
  "name": "Mira",
  "look": "dark curly hair, freckles, 35mm film grain, muted colour",
  "voice": "playful and confident, teasing rather than pleading",
  "scenes": ["on a fire escape in summer, the city behind her"],
  "topics": ["quitting a job nobody liked"]
}
```

## What isn't configurable

**Every image prompt ends with bounds** fixing the subject as a fictional adult
resembling no real person. **A persona mentioning a minor, or asking for
explicit content, is refused** — not warned about, refused, when the file
loads and before anything is billed. Appending "adult" to a prompt that asks
otherwise only creates a contradiction, and image models resolve contradictions
however they like.

**Alt text always begins "AI-generated image."** It costs none of the 280
characters and tells anyone using a screen reader, or anyone who checks, what
they're looking at.

This assumes the account says it's automated in its bio. X's rules require it,
and the version that hides it is the version that gets suspended.

## Setting it up

Repository **secrets**:

| | |
|---|---|
| `X_API_KEY`, `X_API_SECRET` | app credentials from developer.x.com |
| `X_ACCESS_TOKEN`, `X_ACCESS_SECRET` | access token for the posting account |
| `CAPTION_KEY` | Groq, OpenRouter, Gemini — anything OpenAI-compatible |
| `IMAGE_KEY` | only if your image provider needs one |

Repository **variables**: `CAPTION_URL`, `CAPTION_MODEL`, `IMAGE_URL`,
`IMAGE_MODEL`. Captions default to Groq's endpoint; `CAPTION_MODEL` has no
default and the bot says so rather than guessing a name that may be retired.
Images default to Pollinations, which needs no key, so the first dry run works
before you sign up for anything.

Run the workflow by hand with **publish** off first. It builds a post, uploads
the image as an artifact, and posts nothing.

## How a caption gets chosen

The model is asked for six. Hard rules drop anything unusable — too short, over
280, hashtags, @-mentions, links, emoji, more than one exclamation mark — and
anything resembling a recent post. `preference` then picks among what's left.

That function is a **preference, not a measurement**: four beliefs about what
reads well, none tested against engagement data. It's forty lines and the
comment says as much, so nobody later mistakes the number for evidence.

## State

`posted.json` holds the last 30 posts. The Actions runner is thrown away each
run, so the workflow commits it back — without that she repeats herself.

## Tests

```bash
python -m pytest tests/ -q
```

72 tests, no network. The OAuth 1.0a signing is pinned against X's own
published test vector, because a wrong signature is a 401 with no useful body:
every other test would pass while nothing could post.

**Nothing here has posted a real tweet.** X was unreachable from where this was
built. The signing is proven against that vector and every guard is tested, but
whether your API tier permits media upload is something only your first
`--post` will tell you.
