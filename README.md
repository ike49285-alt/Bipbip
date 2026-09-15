# thirsttrap

Generates candidate posts for X and ranks them, entirely on your machine. No
API, no network, no key, no third-party packages — Python standard library only.

```bash
python -m thirsttrap chat "finally quitting the job"
```

```
400 candidates, showing top 3  [flirt]

1.  84.8   Nobody tells you about finally quitting the job. It gets quieter.
2.  75.8   You already know about the job. You're just rehearsing it.
3.  73.2   Who told you finally quitting the job needed a reason you like?

> shorter, no questions
> never use exclamation marks
+ standing rule: never say '!'   (/forget to drop it)

> /keep 1
kept (1 total)
learned: direct_address +0.042, length -0.037   (barely tuned (1/20 keeps))
```

## How it works

**Generation is a grammar.** `topic.py` parses what you typed into the forms a
sentence needs — `"finally quitting the job"` becomes phrase, gerund, and the
head noun `"the job"`. `grammar.py` slots those into rhetorical frames (withheld
knowledge, second-person assertion, setup-and-punch, reversal) whose blanks are
filled from persona-tagged lexicons, recursively. One frame yields thousands of
surface forms; a run draws 400 and throws most away.

**Selection is the scorer.** `score.py` rates each candidate 0–100 across eight
weighted components, `rank.py` sorts and discounts near-duplicates, and you see
the top few.

**Refinement is filtering.** The grammar can't be *asked* for a shorter post, so
`directives.py` turns "shorter" into `max_chars=95` and the pool is filtered.
That's why constraints always work — but see the limits below for what it costs.

## Use

```bash
python -m thirsttrap chat "leg day" -p gym      # interactive
python -m thirsttrap gen "quitting" -k 5 -b     # one shot, with breakdown
python -m thirsttrap gen "leg day" --seed 4     # reproducible
python -m thirsttrap score "You already know."  # score your own writing
python -m thirsttrap personas
```

In chat, type a topic to start, then adjustments: `shorter`, `much shorter`,
`longer`, `one line`, `no questions`, `no numbers`, `no emoji`, `more like 2`,
`try deadpan`, `stop using "game changer"`.

| | |
|---|---|
| `/rules`, `/forget` | standing rules, and dropping them |
| `/weights`, `/untune` | what she's learned, and resetting it |
| `/keep N`, `/kept`, `/drop N`, `/save PATH` | collect and write out |
| `/persona NAME`, `/personas` | voice |
| `/pool N`, `/top K`, `/breakdown` | how many drawn, shown, explained |
| `/clear` | drop this turn's adjustments, keep standing rules |
| `/again`, `/profile`, `/help`, `/quit` | |

## She tunes herself as you talk

Two mechanisms, and they're different in kind.

**Rules you state.** Phrase something as a rule — "never use exclamation marks",
"always one line", "from now on, shorter" — and it's parsed into a standing
constraint saved to `~/.config/thirsttrap/profile.json`. Anything phrased as a
one-off applies to the current topic only. New rules print when adopted;
`/rules` lists them, `/forget` clears them.

**What you keep.** Keeping a post is a statement that it beat the others on
screen, so the component weights shift toward whatever distinguished it. Over
many keeps the ranking stops reflecting the shipped prior and starts reflecting
you. `/weights` shows the drift against the prior, `/untune` reverts it.

```
> /weights
barely tuned (3/20 keeps)
  hook            ######################## 0.211  (prior 0.20, +0.011)
  length          ################........ 0.129  (prior 0.14, -0.011)
  ...
```

The confidence label is not decoration. Eight weights fitted from a handful of
binary choices is badly underdetermined, and the interface says so rather than
letting three keeps look like a model of your taste.

## What the score means, and when it doesn't

The weights are a **prior** — beliefs about what travels on X (hashtags and
links suppress reach, second person outperforms third), not a fit to engagement
data. Practical range is about 40–85, not 0–100; read gaps between candidates,
not absolute numbers.

**On generated posts the score is the objective, not a judgment.** Picking the
top of a pool by score guarantees a high score the way picking the tallest
person in a room guarantees height. It's still the right way to choose — it just
isn't evidence the post is good, and a rising score across a session mostly
means the search is working.

The score only acts as an outside opinion on text the grammar didn't write,
which is what `thirsttrap score` is for. That command is the honest one.

## Limits

Worth knowing before you expect too much:

- **It doesn't understand your topic.** It arranges your words inside shapes
  that read well. Give it something the frames can't hold and you'll get
  grammatical nonsense.
- **Output is more formulaic than a language model's.** Same frames recur; the
  mitigation is volume plus the near-duplicate penalty, not cleverness.
- **Directives are keyword matching.** A fixed vocabulary is recognised and
  everything else is treated as a new topic. That's deliberate — guessing at an
  unrecognised sentence is worse than ignoring it — but it means phrasing
  matters more than it would with a model.
- **A constraint that nothing satisfies returns nothing** rather than quietly
  relaxing itself. `/clear` to back out.
- Text only. It doesn't post anything, and there's no X integration.

## Tests

```bash
python -m pytest tests/ -q
```

266 tests. No network, no mocks, no fakes — everything is local and
deterministic under a fixed seed, so the tests exercise the real code paths.
