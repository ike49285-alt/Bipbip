# thirsttrap

Generates candidate posts for X with Claude, then ranks them by a scoring prior
so you pick from the top of a batch instead of the first thing that came out.

Works one-shot from the command line, or as a conversation you refine in place.

Two halves that deliberately don't talk to each other:

- **`generate.py`** asks Claude for N genuinely different takes on a topic in a
  chosen voice.
- **`score.py`** scores any text 0–100 on eight weighted components. It needs no
  API access and no network, so it works on posts you wrote yourself.

`rank.py` joins them and discounts candidates that repeat each other.

## Install

```bash
pip install -r requirements.txt
```

Generation needs Claude API credentials — `ANTHROPIC_API_KEY`,
`ANTHROPIC_AUTH_TOKEN`, or an `ant auth login` profile. Scoring needs none.

## Use

```bash
# Talk to it -- the first thing you say is the topic, the rest is refinement
python -m thirsttrap chat "finally quitting the job"

# Generate 12 candidates, show the best 3
python -m thirsttrap gen "finally quitting the job" -n 12 -k 3

# Pick a voice
python -m thirsttrap gen "leg day" -p gym -b

# Score posts you already have — no API access needed
python -m thirsttrap score "You already know the answer. You're just waiting."

# Rank a file of drafts
cat drafts.txt | python -m thirsttrap score -b

python -m thirsttrap personas
```

`-b/--breakdown` prints the per-component bars, which is the only way to tell
*why* something ranked where it did.

## Chat mode

```
> finally quitting the job
3 candidates, showing top 3  [flirt]

1.  82.8
   Nobody tells you the best part of quitting. The silence after.
2.  77.1
   You already know the answer. You're just waiting for permission.

> shorter, more like 2
1.  76.0
   You already know. You're stalling.
2.  61.4
   Quit at 4. Home by 5. Never felt lighter.
3.  45.0  [near-duplicate]
   Quit at 4. Home by 5. Never felt better.

> /keep 1
kept (1 total)
```

The conversation is real multi-turn context, so "shorter", "less earnest", or
"more like 2" all resolve against what came before.

**The numbering is shared.** After each batch is ranked, it goes back into the
conversation numbered in the order you saw it — so post 2 on your screen is post
2 to the model. Rank first and number second, or every reference points at the
wrong post.

| | |
|---|---|
| `/persona NAME` | switch voice (`/personas` lists them) |
| `/n N`, `/top K` | batch size, how many to show |
| `/keep N`, `/kept`, `/drop N` | pin candidates across turns |
| `/save PATH` | write pinned posts to a file |
| `/score TEXT` | score text locally, costs no request |
| `/breakdown` | toggle component bars |
| `/again` | re-run the last instruction |
| `/reset` | forget the conversation, keep the pins |

The kept list survives `/reset` and persona switches, so you can collect across
several directions and save at the end. Commands are handled locally — only
plain text costs an API call.

## The eight components

| Component | Weight | Rewards |
|---|---|---|
| `hook` | 0.20 | The first eight words — second person, a number, a contrarian opener. Penalises hedged openings (`I just think maybe…`). |
| `length` | 0.14 | The 60–140 character band. Zero above 280. |
| `direct_address` | 0.12 | `you` / `your`, one to three times. More reads as nagging. |
| `curiosity` | 0.12 | One question mark, open loops, an unresolved referent. |
| `restraint` | 0.12 | Starts at 1.0 and subtracts for hashtags, links, 3+ emoji, multiple exclamations, shouting. |
| `concreteness` | 0.10 | Numerals and named things. Penalises `-ness` / `-ity` / `-tion` abstraction. |
| `rhythm` | 0.10 | Setup then a short punch. Penalises four-plus sentences. |
| `freshness` | 0.10 | Absence of worn-out phrases (`let that sink in`, `game changer`, `read that again`). |

Novelty is handled separately, at rank time, because it's a property of the
batch rather than the post: candidates are walked best-first and each is
discounted by its word overlap with the better ones above it. The strongest
member of a near-duplicate cluster keeps its full score; its echoes pay.

## What this score is and isn't

**It's a prior, not a measurement.** Every weight encodes a belief about what
travels on X. Those beliefs come from platform behaviour that's widely reported
— hashtags and off-platform links suppress reach, second person outperforms
third — not from engagement data this package has fitted. Nothing here has been
validated against your account, or anyone's.

So read it as a way to order a batch against itself, not as a prediction of
likes. Two specific cautions:

- **The practical range is about 40 to 85, not 0 to 100.** Across a spread of
  deliberately good and deliberately terrible posts, the worst scored 41 and the
  best 84. A 52 is bad. Read the gaps between candidates, not the absolute
  number.
- **`restraint` is partly pre-satisfied.** The generator is told not to use
  hashtags or links, because those are platform facts worth stating up front. So
  generated candidates nearly always score 1.0 there, and that component earns
  its keep on text *you* wrote, not on text the tool produced.

Everything else is kept out of the generation prompt on purpose. If the model
were told the rubric, the score would mostly measure how well it followed
instructions it had just been handed, and every batch would look excellent.
`test_generate.py` asserts the rubric doesn't leak into the prompt.

## Calibrating this

Replacing the prior with a fit needs data the package doesn't have: your own
posts and their impressions. The shape of that work:

1. Export your posts with their impression and engagement counts.
2. Score each one with `score_post` and keep the component vector.
3. Regress engagement rate — engagements per impression, not raw likes, or
   you'll just rediscover your follower growth — on the eight components.
4. Replace `WEIGHTS` with the fitted coefficients.

Two things will bite. Posting time and follower count swamp text effects, so
they belong in the regression as controls even though you can't act on them.
And a few hundred posts is a small sample for eight predictors — expect the
fitted weights to move a lot under resampling, and check that they do before
trusting them.

## Limits

- Text only. It doesn't generate, edit, or evaluate images.
- It doesn't post anything. There's no X API integration and no scheduler.
- It won't write sexually explicit content, won't write as a real named person,
  and treats all subjects as adults.

## Tests

```bash
python -m pytest tests/ -q
```

128 tests, no network calls — the generator and chat tests drive a fake client.
