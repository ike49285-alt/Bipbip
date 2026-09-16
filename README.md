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
python -m bot check              # is persona.json sane?
python -m bot prompts -n 20      # prompts to paste into a generator
python -m bot caption out/       # caption those images into the timeline
python -m bot                    # open the studio: chat + timeline + tuning
```

`persona.json` is the only thing you edit. Everything reads from it — the image
prompts, the caption voice, the conversation, the boundaries.

## The studio

`python -m bot` opens a local window on port 8000 with three panes:

- **Chat** — talk to her. Each message is annotated with the boundary it
  tripped, if any.
- **Timeline** — what has been captioned so far.
- **Tune** — edit her voice rules and facts and see the change take effect on
  the next message. The panel shows the exact system prompt being sent. Edits
  only touch `persona.json` if you tick the box.

Set `ANTHROPIC_API_KEY` for replies. Without one she stays quiet, but every
message is still classified and annotated — which is the half you can tune for
free.

## Generating images

`bot prompts` emits prompts; where you run them is up to you. For the full
pipeline there are two Kaggle notebooks: `notebooks/bootstrap.py` generates
from the anchor and curates a training set, then `notebooks/train_lora.py`
trains the LoRA on it.

| | Good for | Not for |
|---|---|---|
| **Perchance** | Finding an anchor face — browser, zero setup | Anything repeatable |
| **Cloudflare / Pollinations** | Generic images, quick tests | Her — hosted APIs can't run your model |
| **Kaggle notebook** | The real pipeline: IP-Adapter, LoRA, batches | Anything on-demand |

Free hosted APIs can't hold a character: they serve *their* models from *your
text prompt*, with nowhere to put an anchor face or a trained LoRA. That's why
`notebooks/bootstrap.py` targets a Kaggle GPU — see `DESIGN.md`.

## What's built

| | |
|---|---|
| Persona core, voice card, validation | done |
| Prompt builder | done |
| Identity gate + curation (`bot/qc.py`) | done |
| Bootstrap orchestration (`bot/bootstrap.py`) | done, tested |
| Bootstrap notebook | thin driver, **diffusers calls never executed** |
| LoRA training wrapper | written, **never executed** |
| Captioning + repetition scoring | done |
| Chat agent + boundaries | done |
| Studio UI + CLI | done |
| ~~X integration~~ | dropped |

222 tests. Standard library only, except `anthropic` for captions and chat.

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
notebooks/         Kaggle-side image generation
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
