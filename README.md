# Persona bot

A Twitter/X account for a **disclosed AI persona** — generated selfies,
captions, and DM replies — built to run on free infrastructure.

Read [DESIGN.md](DESIGN.md) for the architecture, the free-hosting constraints,
and the calculated running costs. The short version: image generation happens in
occasional free-GPU notebook sessions and fills a human-approved queue; GitHub
Actions cron only ever consumes that queue. Ongoing spend is DM tokens alone,
a few dollars a month.

**Preview:** https://claude.ai/artifact/4zvEoi2b4Xfi2F3o8V2qTE — timeline and DM
gates, with real prompts and hand-written target captions. Private by default.

The preview's source is deliberately *not* committed: it embeds the anchor as
base64, so committing it to a public repo would publish that face through the
back door, which is still an open decision.

## Status

| Step | |
|---|---|
| 1. Persona core + voice card | done |
| 2. LocalDriver + local UI | done |
| 3a. Prompt builder (`bot/scenes.py`) | done |
| 3b. QC gate + curation (`bot/qc.py`) | done |
| 3b. Bootstrap notebook (`notebooks/bootstrap.py`) | written, **unverified** |
| 3b. LoRA training | not started — pin a trainer first |
| 4. Captioning + anti-repetition (`bot/captions.py`) | done |
| 5. DM agent + gates | not started |
| 6. Approval queue + Actions workflows | not started |
| 7. XDriver | not started |

## Running it

No dependencies — standard library only.

```bash
python -m bot.ui            # http://127.0.0.1:8000
python -m bot.scenes -n 10  # prompts to paste into a generator
python -m bot.scenes -n 200 --json > content/manifest.json   # for the notebook
```

Then run `notebooks/bootstrap.py` on a Kaggle GPU to generate against the
manifest and filter through the QC gate. Sweep `IP_ADAPTER_SCALE` on one
prompt before committing to a full batch.

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
bot/qc.py        identity gate + curation; embedding backend is injected
bot/captions.py  vision pass, voice pass, repetition scoring; LLM injected
bot/ui.py        local timeline and DM inbox, stdlib http.server
```

`bot/persona.py` will refuse to load a persona that doesn't disclose itself as
synthetic. That's deliberate — see the ground rule in DESIGN.md.
