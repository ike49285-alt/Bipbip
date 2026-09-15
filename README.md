# thirsttrap

A writing partner that runs on your own machine. You talk to her about what
happened; she talks back, and drafts posts when there's something worth
drafting. A local LLM writes; a scorer that learns from what you keep decides
what to show you. No hosted API, no API key, nothing leaves your machine.

```bash
ollama pull llama3.2
python -m thirsttrap chat "finally quitting the job"
```

```
thirsttrap [flirt] -- untuned
talking to ollama llama3.2 @ http://127.0.0.1:11434

> ugh I finally quit today
about time. how did it feel walking out?

> honestly it felt kind of anticlimactic
anticlimactic is the post. that's the whole thing.

1.  84.8   Nobody tells you the best part of quitting. It's how quiet 4pm gets.
2.  77.1   You already know the answer. You're just waiting for permission.
3.  57.8   Three years, and the walk to the lift took ninety seconds.

> never use exclamation marks
+ standing rule: never say '!'   (/forget to drop it)

> /keep 1
kept (1 total)
learned: direct_address +0.042, length -0.037   (barely tuned (1/20 keeps))
```

**She is not a form that takes a topic and returns tweets.** The first turn
above produced no drafts on purpose — she didn't have the detail yet, so she
asked for it. A turn where she only talks is a normal turn. The conversation is
real: her replies and her drafts go back to her on the next turn, so "more like
2" and "shorter than that" resolve against what actually happened.

## Backends

Probed in order; the first one running wins. Override with `--backend`.

| Backend | What it is | Setup |
|---|---|---|
| `ollama` | Ollama's API on `127.0.0.1:11434` — the default | `ollama pull llama3.2` |
| `openai-compat` | llama.cpp's `llama-server`, LM Studio, vLLM — anything serving `/v1/chat/completions` | `llama-server -m model.gguf` |
| `llama-cpp` | A GGUF loaded in-process, no server at all | `pip install llama-cpp-python`, then `--gguf path.gguf` |
| `grammar` | No model — a slot grammar fallback | nothing |

```bash
python -m thirsttrap chat "leg day" --backend ollama --model mistral
python -m thirsttrap chat "leg day" --backend openai-compat --host http://127.0.0.1:8080
python -m thirsttrap chat "leg day" --backend llama-cpp --gguf ~/models/qwen.gguf
```

Or set `THIRSTTRAP_BACKEND`, `THIRSTTRAP_MODEL`, `THIRSTTRAP_HOST`, `THIRSTTRAP_GGUF`.
`/backend` in a session says which one is answering.

**You don't have to name a model.** With Ollama, leaving `--model` off makes it
ask the server what you've actually pulled and use the first one — guessing a
name you never pulled is the most common way this fails, and asking is free.

**The package itself has no dependencies.** HTTP to the local server is
`urllib` from the standard library; `llama-cpp-python` is an optional extra
needed only by the in-process backend.

**If no model is found it falls back to the grammar** — a slot grammar over
rhetorical frames that needs nothing installed. It's genuinely worse: more
formulaic, doesn't understand your topic, just arranges your words in shapes
that read well. It exists so the tool still runs, and it says so in the header.

## Two ways to talk to it

```bash
python -m thirsttrap serve        # web UI at 127.0.0.1:8765, your local model
python -m thirsttrap chat "leg day"               # terminal
```

`serve` binds loopback on purpose: it is an unauthenticated endpoint that
drives a language model. `--host-bind` opens it wider, deliberately.

## No computer? Use the browser version

The Pages site converses on a phone with nothing installed and nothing hosted.
Open it, tap **Connect a model** in the sidebar, and paste a free API key:

| Provider | Free key | Suggested model |
|---|---|---|
| OpenRouter | openrouter.ai/keys | `meta-llama/llama-3.3-70b-instruct:free` |
| Groq | console.groq.com/keys | `llama-3.3-70b-versatile` |
| Google Gemini | aistudio.google.com/apikey | `gemini-2.0-flash` |
| Cerebras | cloud.cerebras.ai | `llama3.1-8b` |

**The key stays on your device.** It lives in that browser's `localStorage`,
is sent only to the provider you picked, and is never committed or uploaded.
The page is public, but there is no shared secret in it — anyone else who
opens it brings their own key, or gets the grammar.

Any OpenAI-compatible endpoint works via **Custom**, including Ollama on a
machine you control. Without a key the page still drafts, using the grammar —
it just can't converse.

One caveat I could not test from here: a provider has to allow browser
requests (CORS). OpenRouter and Gemini are built for it; if one refuses, the
page says so by name and you can switch providers in the same panel.

## In a browser

The whole system also runs as a static page, no install and nothing billed:

- **GitHub Pages** — <https://ike49285-alt.github.io/Bipbip/>
- **Claude artifact** — <https://claude.ai/artifact/NPNgsJeHY32B3NvqiGkyLZ>

`index.html` is generated from `web/thirsttrap.html` by
`scripts/build_pages.py`, because the artifact source deliberately has no
`<!doctype>` or `<head>` (the artifact viewer supplies its own). Edit the
source, then:

```bash
python scripts/build_pages.py           # regenerate index.html
python scripts/build_pages.py --check   # CI-style staleness check
```

`test_pages.py` fails if the two drift apart.

The browser version is the whole system ported
to JavaScript: topic parsing, the slot grammar, the scorer, the ranker, the
directive parser and the weight learning, with the profile in `localStorage`.

**It spends nothing and talks to nothing.** Generation is the grammar, in the
page — no model, no network call, no tokens, no account. That is a limit of
the medium rather than a choice of engine: a static or sandboxed page can
neither reach a model on your machine nor download model weights.

The **Claude artifact** copy cannot converse at all — the artifact viewer
blocks every outbound request, so it hides the connect panel and runs the
grammar. The **Pages** copy has no such restriction, which is why the
bring-your-own-key flow above lives there.

## Use

```bash
python -m thirsttrap chat "leg day" -p gym       # interactive
python -m thirsttrap gen "quitting" -n 12 -k 5   # one shot
python -m thirsttrap score "You already know."   # score your own writing
python -m thirsttrap personas
```

Just talk to her. Adjustments are things you say, not flags: `shorter`,
`much shorter`, `one line`, `no questions`, `no emoji`, `more like 2`,
`try deadpan`, `never use exclamation marks`, `stop using "game changer"`.
Phrase one as a rule ("never", "always", "from now on") and it sticks.

| | |
|---|---|
| `/rules`, `/forget` | standing rules, and dropping them |
| `/weights`, `/untune` | what she's learned, and resetting it |
| `/keep N`, `/kept`, `/drop N`, `/save PATH` | collect and write out |
| `/persona NAME`, `/personas`, `/backend` | voice, and which model is answering |
| `/n N`, `/top K`, `/breakdown`, `/clear` | batch size, display, reset this turn |
| `/again`, `/profile`, `/help`, `/quit` | |

## She tunes herself as you talk

Two mechanisms, different in kind, both persisted to
`~/.config/thirsttrap/profile.json`.

**Rules you state.** Phrase something as a rule — "never use exclamation
marks", "from now on, shorter" — and it's parsed into a standing constraint
that goes into every later prompt. One-off phrasing applies to the current
topic only. New rules print when adopted; `/rules` lists, `/forget` clears.

**What you keep.** Keeping a post does two things: it joins the few-shot
examples in the system prompt, and it shifts the ranking weights toward
whatever distinguished it from the others on screen. `/weights` shows the
drift against the shipped prior, `/untune` reverts.

The confidence label — "barely tuned (3/20 keeps)" — is not decoration. Eight
weights fitted from a handful of binary choices is badly underdetermined, and
the interface says so rather than letting three keeps look like a model of your
taste.

**Constraints are applied twice on purpose:** asked for in the prompt, then
enforced by filtering. Small local models ignore instructions often enough that
asking alone doesn't work, and filtering alone wastes most of a slow batch.

## What the score means

The weights are a **prior** — beliefs about what travels on X (hashtags and
links suppress reach, second person outperforms third), not a fit to engagement
data. Practical range is roughly 40–85, not 0–100; read gaps between
candidates, not absolute numbers.

The generator is never told the rubric. If it were, the score would measure how
well the model followed instructions it had just been handed, and every batch
would look excellent. Format rules (length, no hashtags, no links) *are* stated,
because they're platform facts — so `restraint` and `length` are partly
pre-satisfied on generated text and earn their keep on text you wrote yourself.

One caveat that applies only to the grammar fallback: selecting the top of a
400-candidate pool *by score* makes the score the objective rather than a
judgment. With a model generating, the scorer is an independent judge again.

## Limits

- **Small models are small.** A 7B will produce flatter copy and a duller
  conversation than a hosted frontier model, and will sometimes drift off the
  JSON it was asked for — when that happens the whole reply is kept as
  conversation and the drafts are lost for that turn, because losing what she
  said is worse.
- **Nothing here has met a real Ollama.** The protocol, history, JSON mode,
  model auto-detection and error surfacing are all tested against a loopback
  server speaking Ollama's wire format, but this was built where
  `registry.ollama.ai` is unreachable, so no actual model has ever answered.
  Your first `ollama pull` is the real test.
- **Directives are keyword matching**, not understanding. A fixed vocabulary is
  recognised; anything else is treated as a new topic. Deliberate — guessing at
  an unrecognised sentence is worse than ignoring it.
- **A constraint nothing satisfies returns nothing** rather than quietly
  relaxing. `/clear` backs out.
- Text only. It doesn't post anything.

## Tests

```bash
python -m pytest tests/ -q
```

351 tests, no outbound network. Both HTTP surfaces are tested against real
loopback servers — the model backends against one speaking Ollama's and the
OpenAI-compatible protocol, and the web UI against its own — so sockets,
timeouts, error handling and parsing are genuinely exercised rather than
mocked.
