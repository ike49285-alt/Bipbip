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

## The terminal

<https://ike49285-alt.github.io/Bipbip/> — a terminal for her, from a phone.

```
> preview
scene   on a fire escape in summer, the city behind her
topic   the walk home at 3am

  You already know the answer. You're just stalling.

50/280
[image]

> post
dispatched. Actions is building and posting it now.
```

It writes the caption and draws the picture in the page, so previewing costs
nothing and takes seconds. `post` does **not** post from the browser — X's API
refuses browser requests, and OAuth 1.0a from a page would hand your tokens to
anyone who opened it. Instead it dispatches the Actions workflow, which posts
from GitHub with the secrets that never leave the repo.

`help` lists everything. `key gsk_…` for captions, `gh <token>` (needs
`actions:write`) for posting. Both stay in that browser's storage on that
device — not in the repo, not in the page, not sent anywhere but the provider
they belong to.

The persona edits there are local to the phone and drive the preview.
`persona.json` in the repo is what the scheduled post uses; keep them in step
by hand, or treat the terminal as the sketchpad and the file as the decision.

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

76 tests, no network. The OAuth 1.0a signing is pinned against X's own
published test vector, because a wrong signature is a 401 with no useful body:
every other test would pass while nothing could post.

**Nothing here has posted a real tweet.** X was unreachable from where this was
built. The signing is proven against that vector and every guard is tested, but
whether your API tier permits media upload is something only your first
`--post` will tell you.
