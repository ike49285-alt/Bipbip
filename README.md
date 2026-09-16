# Persona bot

A Twitter/X account for a **disclosed AI persona** — generated selfies,
captions, and DM replies — built to run on free infrastructure.

Read [DESIGN.md](DESIGN.md) for the architecture, the free-hosting constraints,
and the calculated running costs. The short version: image generation happens in
occasional free-GPU notebook sessions and fills a human-approved queue; GitHub
Actions cron only ever consumes that queue. Ongoing spend is DM tokens alone,
a few dollars a month.

## Status

| Step | |
|---|---|
| 1. Persona core + voice card | done |
| 2. LocalDriver + local UI | done |
| 3a. Prompt builder (`bot/scenes.py`) | done |
| 3b. Anchor → LoRA → QC gate | needs a GPU session |
| 4. Captioning with anti-repetition | not started |
| 5. DM agent + gates | not started |
| 6. Approval queue + Actions workflows | not started |
| 7. XDriver | not started |

## Running it

No dependencies — standard library only.

```bash
python -m bot.ui            # http://127.0.0.1:8000
python -m bot.scenes -n 10  # prompts to paste into a generator
```

You get a fake timeline and a DM inbox backed by SQLite. Inbound messages are
recorded; nothing replies automatically until step 5.

```bash
pip install pytest && python -m pytest tests/ -q
```

## Layout

```
persona.json     identity, voice rules, facts, hard bounds
bot/persona.py   loads and validates the above; renders the voice card
bot/driver.py    SocialDriver protocol + LocalDriver (SQLite)
bot/scenes.py    scene pools → generator-ready prompts (backend-independent)
bot/ui.py        local timeline and DM inbox, stdlib http.server
```

`bot/persona.py` will refuse to load a persona that doesn't disclose itself as
synthetic. That's deliberate — see the ground rule in DESIGN.md.
