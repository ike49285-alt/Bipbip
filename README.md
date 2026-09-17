# Character studio

A local tool for building and tuning a disclosed AI persona: generate images of
her, caption them in her voice, and talk to her. Everything runs on your
machine. There is no posting to any platform.

The persona shipped here is **Remy** (`@remy_synthetic`) — an agender femboy,
she/her, drawn rather than photographed, and labelled as AI in her own bio.

**Preview:** https://claude.ai/artifact/4zvEoi2b4Xfi2F3o8V2qTE

## The loop

```
persona.json ──▶ bot prompts ──▶ [your generator] ──▶ bot caption ──▶ timeline
     ▲                                                                   │
     └──────────────── bot chat: talk to her, edit her voice ◀────────────┘
```

Four commands, one file of truth.

```bash
python -m bot doctor             # what can this machine run?
python -m bot generate -n 20     # render her pictures on your GPU
python -m bot curate             # score them, pick a LoRA training set
python -m bot train              # train her LoRA
python -m bot caption out/       # caption those images into the timeline
python -m bot direct "too whiny" # tell the writer what's wrong with the captions
python -m bot                    # open the studio: chat + timeline + tuning
```

## Directing the voice

You don't hand-edit rules to change how she writes — you say what's wrong with
what she wrote:

```
$ python -m bot direct "too whiny"
cut the self-pity; the annoyance is stated once and dropped
  + no self-pity: name the annoyance once, then move on
  + never end on a complaint
  - sentence fragments over full sentences
```

The note becomes a *concrete, checkable* rule in `persona.json` — "no
self-pity, name it and move on", never "be less whiny". Adjectives in a voice
card give the writer nothing to follow. Everything downstream reads the revised
file.

The published preview does the same thing in a browser, against a fixed set of
scenes so a change you make is a change you can see.

`persona.json` is the only thing you edit. Everything reads from it — the image
prompts, the caption voice, the conversation, the boundaries.

## The studio

`python -m bot` opens a chat window on port 8000. You talk to her; that's the
whole screen. Three links underneath open the rest:

- **Edit her voice** — her rules and facts, as plain text. Change them, talk
  again, see the difference. Nothing is written to `persona.json` unless you
  tick "keep them".
- **Her captions** — what has been captioned so far.
- **Start over** — clears the conversation.

Replies where a boundary fired carry a short note under them saying why.

Set `ANTHROPIC_API_KEY` for replies. Without one she stays quiet, but every
message is still classified and annotated — which is the half you can tune for
free.

## Generating images

Run these on your own GPU, or on a free Kaggle one. Same commands either way —
`notebooks/kaggle.py` just clones the repo and calls them. Start with
`bot doctor`; VRAM decides the model family and it picks automatically.

| VRAM | Family | Resolution |
|---|---|---|
| 7GB+ (Kaggle T4/P100, 16GB) | SDXL | 1024px |
| under 7GB (GTX 1060, 6GB) | SD 1.5 | 512px |

**Kaggle is worth the round trip.** 30 GPU-hours a week, free, on a card big
enough for SDXL — better pictures than a 6GB local card can make, and faster.
The cost is that it's batch work: upload the anchor as a private dataset, run,
download the LoRA. Sessions cap at about 9 hours and can drop, but `generate`
resumes from whatever is already on disk.

Once the LoRA exists, bring it home — generating *with* a trained LoRA is much
lighter than training one, and runs fine on the 1060.

The full run:

```bash
python -m bot generate --sweep        # one prompt at four anchor strengths
python -m bot generate -n 200 --scale 0.65
python -m bot curate --keep 40        # score against the anchor, pick a set
python -m bot train                   # LoRA, on the curated set
python -m bot generate --lora content/lora/*.safetensors
```

**Sweep first.** `--scale` is how hard the anchor pulls: too high and every
picture copies its pose, too low and she stops being recognisably herself. That
one number decides whether the batch is usable, and four test renders is
cheaper than finding out after two hundred.

`generate` resumes — a crashed or interrupted run picks up where it stopped.

Free hosted image APIs can't help here: they serve *their* models from *your
text prompt*, with nowhere to put an anchor face or a trained LoRA. That's the
whole reason this runs locally.

## What's built

| | |
|---|---|
| Persona core, voice card, validation | done |
| Prompt builder | done |
| Identity gate + curation (`bot/qc.py`) | done |
| Image pipeline (`bot/images.py`, local) | written, **never executed here — no GPU** |
| Batch orchestration (`bot/bootstrap.py`) | done, tested |
| LoRA training wrapper | written, **never executed** |
| Captioning + repetition scoring | done |
| Chat agent + boundaries | done |
| Studio UI + CLI | done |
| ~~X integration~~ | dropped |

256 tests. Standard library only, except `anthropic` for captions and chat.

```bash
pip install pytest && python -m pytest tests/ -q
```

## Layout

```
persona.json       identity, voice rules, facts, hard boundaries
bot/persona.py     loads and validates it; renders the voice card
bot/scenes.py      scene pools → generator-ready prompts
bot/qc.py          identity gate + curation (embedding backend injected)
bot/captions.py    vision pass, voice pass, repetition scoring (LLM injected)
bot/chat.py        inbound classification, outbound gate, canned fallbacks
bot/driver.py      local storage for the timeline and conversation
bot/ui.py          the studio server
bot/ui.html        its markup — edit without touching Python
notebooks/kaggle.py  runs the same commands on a free Kaggle GPU
```

## Two rules the code enforces

**She discloses.** `Persona.load()` refuses a persona that doesn't identify
itself as synthetic. The chat agent re-checks every outgoing reply: a model that
dodges "are you real?" or claims to be human is overridden, and her own
disclosure line is sent instead. That's a guarantee, not a prompt instruction.

**Boundaries are deterministic.** Money, personal details, meetups, and any
signal of a minor are matched by string rules that hold regardless of which
model sits behind them — matched both plainly and squashed, so `c a s h a p p`
lands the same as `cashapp`. A minor signal ends the thread without calling the
model at all.

## The anchor

`content/anchor/remy-anchor.jpg` is the one reference picture of her.
Generators have no memory between runs — the same prompt twice gives you two
different people — so the only way to get *her* repeatedly is to hand the
generator this image and say "this person, new scene."

```
anchor ──▶ ~200 varied images ──▶ curate 40 ──▶ LoRA ──▶ unlimited on-model images
```

Everything inherits from it, and nothing regenerates it. Lose the anchor and
you pick a new face and start over; nothing new will match what you already
have.

It is **gitignored and not committed**, because committing it to a public repo
publishes that face permanently. That's the reversible default — but it means
the file exists only where you put it. Back it up, or commit it deliberately.

For a Kaggle run, upload it as a private dataset: a fresh clone of this repo
won't contain it.
