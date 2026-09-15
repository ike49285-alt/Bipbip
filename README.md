# thirsttrap

Generates candidate posts for X with a **local** LLM, then ranks them by a
prior that learns from what you keep. No hosted API, no API key, nothing leaves
your machine.

```bash
ollama pull llama3.2
python -m thirsttrap chat "finally quitting the job"
```

```
thirsttrap [flirt] -- untuned
generating with ollama llama3.2 @ http://127.0.0.1:11434

> finally quitting the job
12 candidates, showing top 3  [flirt]

1.  84.8   Nobody tells you the best part of quitting. The silence after.
2.  77.1   You already know the answer. You're just waiting for permission.

> shorter, no questions
> never use exclamation marks
+ standing rule: never say '!'   (/forget to drop it)

> /keep 1
kept (1 total)
learned: direct_address +0.042, length -0.037   (barely tuned (1/20 keeps))
```

## Backends

Probed in order; the first one running wins. Override with `--backend`.

| Backend | What it is | Setup |
|---|---|---|
| `ollama` | Ollama's API on `127.0.0.1:11434` | `ollama pull llama3.2` |
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

**The package itself has no dependencies.** HTTP to the local server is
`urllib` from the standard library; `llama-cpp-python` is an optional extra
needed only by the in-process backend.

**If no model is found it falls back to the grammar** — a slot grammar over
rhetorical frames that needs nothing installed. It's genuinely worse: more
formulaic, doesn't understand your topic, just arranges your words in shapes
that read well. It exists so the tool still runs, and it says so in the header.

## Use

```bash
python -m thirsttrap chat "leg day" -p gym       # interactive
python -m thirsttrap gen "quitting" -n 12 -k 5   # one shot
python -m thirsttrap score "You already know."   # score your own writing
python -m thirsttrap personas
```

In chat, type a topic, then adjustments: `shorter`, `much shorter`, `longer`,
`one line`, `no questions`, `no numbers`, `no emoji`, `more like 2`,
`try deadpan`, `stop using "game changer"`.

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

- **Small models are small.** A 7B will produce noticeably flatter copy than a
  hosted frontier model. The ranker helps by throwing most of a batch away, but
  it can't add wit that isn't there.
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

337 tests, no network. The HTTP backends are tested against a real loopback
server speaking both protocols — actual sockets, not mocks — so the urllib
paths, timeouts, error handling and response parsing are genuinely exercised.
