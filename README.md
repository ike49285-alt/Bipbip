# synth

Tooling for running a **disclosed** synthetic creator — a character who is
openly an AI, and says so everywhere it matters.

## Why disclosed

This is the whole design constraint, so it goes first.

A synthetic character who is open about what she is, is legal, welcome on every
major platform, and a real business. Neuro-sama streams to hundreds of
thousands of people who all know exactly what they are watching; being an AI is
the *draw*. Lil Miquela has taken millions in brand deals without ever claiming
to be human.

A synthetic character passing as human is fraud against whoever is paying for
it. It also ends in a terminated account that takes the operator's other
accounts with it, because the platforms that host paid creator content verify
real, consenting humans.

So the disclosure is not a disclaimer bolted on at the end. It is the load-
bearing wall, and it lives in code:

- `synth/disclosure.py` holds rules merged into **every** persona at load time.
  A spec cannot remove one — there is no syntax for it.
- Every published caption carries a marker, checked twice: once in the content
  gate, once more at the platform on the way out. Both refuse rather than fix.
- A persona whose bio does not make clear what it is **will not load**.
- 38 tests try to route around all of that.

## What is here

```
synth/disclosure.py   the non-negotiable rules + "are you real?" detection
synth/persona.py      the one spec everything else is rendered from
synth/builder.py      draft a new character from a one-line brief
synth/roster.py       several characters at once - and whether they are alike
synth/canon.py        append-only record of what has been established in public
synth/content.py      idea -> draft -> canon check -> disclosure check -> queue
synth/models.py       open-weights backends + the cleanup their output needs
synth/platforms/      publishing adapters; DryRun is the default
personas/example.yaml a complete spec to copy
```

## Running more than one

The hard part of a roster is not writing six specs. It is that the sixth comes
out as the first one with different hair - same sentence rhythm, same six
adjectives, same three opinions - and anyone following two of them notices in a
week. Each spec reads fine alone; together they read as output.

That is measurable, so it is measured. Characters are distinct to a reader
mostly through VOICE: the words they reach for, the words they will not touch.
`synth roster` scores every pair on vocabulary overlap, weights voice double
(two characters can care about the same things and stay distinct; they cannot
talk the same way and stay distinct), and names the pair to fix first.

```
  pair                                 voice  interest   score
  Sophie / Mira                         1.00      0.50    0.83   TOO ALIKE
      shared: actually, cutting, iconic, literally, precise, problem
  Sophie / Wren                         0.13      0.25    0.17
  Sophie / Juno                         0.00      0.00    0.00
```

The threshold is a judgement call, so it is a parameter and it is printed on
every report. A shared handle or name is not a similarity, it is a collision,
and those are reported as errors.

## The two real problems this solves

**Drift.** A character becomes real to an audience through accumulated
consistency, and dies when she contradicts herself. That happens when the
character lives in a dozen prompts that were each edited separately. Here she
lives in one file, and every prompt, bio and caption is rendered from it.

**Contradiction.** A model has no memory between calls, and a prompt stuffed
with "remember that you said" degrades everything else in it. So canon lives
outside the model: an append-only log of what has been said in public, queried
for the handful of facts relevant to the current draft, and checked afterwards.
Retcons are recorded rather than applied — the audience saw the original, and
pretending otherwise loses the people paying closest attention.

## Use

Runs end to end with no API key. The default completer returns an obvious
placeholder rather than plausible text, so nothing fake gets mistaken for real
output.

```bash
python3 -m synth validate personas/example.yaml
python3 -m synth prompt   personas/example.yaml
python3 -m synth canon    personas/example.yaml --add "Vee likes rain"
python3 -m synth canon    personas/example.yaml --check "hated the rain today"
python3 -m synth draft    personas/example.yaml --idea "a playlist" \
                          --text "made a playlist for rainy afternoons"

python3 -m synth new      --name Sophie --handle @sophie \
                          --brief "rates your taste and is unkind about it" \
                          --out personas/sophie.yaml
python3 -m synth roster   personas/
```

`new` drafts a character from a brief and writes a spec file you then own and
edit. The disclosure block is **injected after the model answers**, never
generated - a model asked for "a sharp, funny character" will happily return
one with no disclosure, or with a hedged one that sounds like a disclosure and
is not. With no model configured it writes a scaffold: structure correct,
character deliberately absent.

## Running it on an open model

Two adapters cover nearly everything: native **Ollama** for a local box, and
any **OpenAI-compatible** `/v1/chat/completions` endpoint — llama.cpp's server,
vLLM, LM Studio, Together, Groq, OpenRouter, DeepInfra. Plain stdlib HTTP, no
extra dependency.

```bash
# local, via Ollama
ollama serve && ollama pull llama3.1:8b
export SYNTH_MODEL=llama3.1:8b
python3 -m synth draft personas/sophie.yaml --idea "a playlist"

# anything OpenAI-compatible
export SYNTH_BACKEND=openai
export SYNTH_BASE_URL=http://localhost:8000
export SYNTH_MODEL=mistral-7b-instruct
export SYNTH_API_KEY=...        # only for hosted providers
```

`SYNTH_TEMP` defaults to **0.9**. A persona is a voice, and a voice at
temperature 0.2 is a press release.

**The output cleanup is not an afterthought.** Ask an instruction-tuned open
model for one post and you get:

```
Sure! Here's a flirty post for Sophie:

"made a playlist for rainy afternoons."

Let me know if you'd like a different tone!
```

Three of those four lines are not the post — and every one of them would sail
through the content gates untouched, because none of them is too long,
contradicts canon, or claims to be human. They would simply go out with the
quote marks still on. So completions are stripped before they reach the
pipeline, and the stripping is the part with tests on it: 21 cases pushing in
both directions, because over-cleaning is worse than under-cleaning. It mangles
a good post into something that still reads like the character, and nobody
notices.

With nothing configured, every command still runs and `draft` refuses to queue
anything — the gates are safety gates, not quality gates, and a placeholder
passes all of them.

Any `str -> str` callable works as a `Completer` if you want a different
backend. The gates do not know or care which model is behind it — that is the
point. The model is swappable; the gates are the product.

## Not done

- No live platform adapters yet. `DryRun` is the only one, deliberately: the
  thing to review is real output over weeks, not a description of it.
- Nothing posts on its own, and nothing should. `approved` means it passed the
  automatic checks; a human releases it. An unattended poster is how an account
  dies at 4am.
- No image or voice generation — still the biggest gap.
- No scheduler or analytics.
