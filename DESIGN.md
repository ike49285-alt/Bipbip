# Character studio — design

A local tool for building and tuning a disclosed AI persona: generated images,
captions in her voice, and conversation. Runs on your machine and on free
infrastructure for the GPU work. No platform integration — that was dropped;
see "No platform integration" below.

## Ground rule

This is a **disclosed** AI persona: labeled in the bio, and the DM layer answers
truthfully when asked whether it's a bot. That is an architectural property, not
a disclaimer — `bot/persona.py` refuses to load a persona without a disclosure
line, and the DM gates treat "are you real?" as an always-answer case.

The distinction matters: a labeled synthetic persona is a legitimate genre. The
same system that hides what it is, in one-on-one DMs, is a catfishing rig. The
code should make the first easy and the second awkward.

## The constraint that shapes everything

Free hosting means **no always-on GPU** and **no always-on server**. The naive
design — generate a selfie on demand, caption it, post it — needs both. So it's
restructured into two halves that run in different places at different times:

```
 ONE-TIME / OCCASIONAL              ONGOING
 free GPU notebook                  GitHub Actions cron
 (Colab · Kaggle)                   (free on public repos)
 ─────────────────────              ────────────────────────
 anchor face                        post job  ─ pick from queue
   → LoRA train                                └ mark used
   → batch generate 200
   → QC gate (auto)                 dm job    ─ poll inbox
   → human curate                             ├ policy gate
   → batch caption                            ├ model reply
   → commit queue ──────────────────┬────────▶└ send
                                    │
                            content/queue.json
                            content/img/*.jpg
```

Generation is decoupled from posting. You refill a queue occasionally on a free
GPU; the bot only ever *consumes* the queue. This is cheaper, and it's also
safer — every image is human-approved before it can be posted.

## Free infrastructure

| Need | Choice | Real limits |
|---|---|---|
| GPU for LoRA + batch gen | Kaggle notebooks | 30 GPU-hr/week, free. More generous than Colab. |
| Scheduled compute | GitHub Actions | Unlimited minutes on **public** repos; 2,000 min/mo private. |
| State | SQLite in-repo, committed | Fine at this volume. Move to Turso/D1 free tier if it grows. |
| Image hosting | The repo, or Cloudflare R2 | R2: 10GB + no egress fees, free. |
| Always-on (if ever needed) | Oracle Cloud Always Free | 4 ARM cores / 24GB, genuinely free, no expiry. |

### Generation backends: why the free APIs don't fit

Free image APIs exist and are real -- Cloudflare Workers AI gives ~230 FLUX
images/day with no credit card and hard-stops instead of billing; Pollinations
needs no key at all. **Neither can run your character.** Hosted endpoints serve
*their* models from *your text prompt*; there is nowhere to upload a trained
LoRA or an anchor face. Cloudflare's `stable-diffusion-v1-5-img2img` is not a
substitute: img2img with low `strength` preserves the entire input frame -- pose,
composition, everything -- and no value of it means "same face, new scene".
That gap is exactly what IP-Adapter FaceID / PuLID / InstantID exist to close,
and running any of them means running your own inference.

So: **free + API + your character is not a combination that exists.** Pick two.

| Backend | Use for | Not for |
|---|---|---|
| Perchance | Anchor hunting -- browser, zero setup | Anything repeatable |
| Cloudflare / Pollinations | Generic images, plumbing tests | The persona -- can't run your model |
| Kaggle notebook | The real pipeline: LoRA, IP-Adapter, batch | Anything on-demand |

Kaggle is still automation, just at *batch* granularity -- its API can trigger a
notebook run, so "refill the queue" is one scripted call, not 200. That is what
the decoupled queue already assumed, so none of the architecture changes.

Cloudflare ships models weekly and retires them without notice; confirm current
model IDs before building against them.

**Known sharp edges, so they don't surprise you later:**

- Actions cron is *best-effort* — delays of 10–30+ min under load are normal. Posting times are approximate. Don't build anything that needs punctuality.
- Scheduled workflows are **auto-disabled after 60 days of repo inactivity**. A monthly dummy commit, or just refilling the queue, keeps them alive.
- Public repo means the persona's images, captions, and queue are public. Decide if that's acceptable before choosing public for the unlimited minutes.
- Committing SQLite creates merge conflicts if two jobs write concurrently. Serialize with an Actions concurrency group.

## Token cost, actually calculated

Using `claude-haiku-4-5` ($1 / $5 per MTok). Figures are **uncached ceilings**;
the persona prefix is byte-identical on every call, so prompt caching cuts the
input side substantially below these.

**Captioning — one-time, not recurring.** Two passes per image (vision → scene
JSON, then scene → caption in voice, 4 candidates). ~6k in / ~500 out per image.
For a 200-image queue: ~1.2M in, ~100k out ≈ **$1.70 once**.

**DMs — the only ongoing cost.** ~3k in / ~150 out per turn ≈ $0.004/turn.

| Volume | Monthly |
|---|---|
| 10 DM turns/day | ~$1.20 |
| 50 DM turns/day | ~$6 |
| 200 DM turns/day | ~$23 |

So the whole thing is a few dollars a month, dominated entirely by DM volume.
If that's still too much, `bot/llm.py` is a single interface with an Ollama
implementation behind it — local Llama/Qwen is $0, at a real cost in voice
quality and, more importantly, in gate reliability. Keep the safety gates
deterministic (see below) precisely so they don't degrade when the model does.

## The persona core

All three features are the same problem — consistency — in three forms. One
object is the source of truth; nothing else holds persona state.

```
                    persona.json
      identity · visual · voice · facts · bounds
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   selfie gen  ──img──▶  captioning     DM agent
   (visual                (voice          (voice +
    consistency)           consistency)    continuity)
```

## (a) Selfies

Claude doesn't generate images; this is Flux or SDXL in a free notebook. The
hard part is **identity lock** — the same face every time. Text prompts alone
will not do it.

1. **Anchor** — generate one canonical face, curate ruthlessly. Everything inherits from it.
2. **Bootstrap** — IP-Adapter / InstantID on the anchor → ~200 varied images → hand-pick the 40 most on-model.
3. **LoRA** — train on those 40. Now identity is repeatable from a prompt.

Then `LoRA + scene prompt + selfie grammar`: arm's-length framing, front-camera
focal length, ordinary lighting, slight handheld blur. A selfie that looks like
a studio portrait reads as wrong — that's the genre's visual language.

**The QC gate is not optional.** Face-embedding similarity (InsightFace/ArcFace)
against the anchor; below threshold → auto-reject. Plus a hand-artifact pass.
Then a human approves what survives. Scene variety comes from a sampler over
`{wardrobe × location × time-of-day × activity}` with a recency penalty.

## (b) Captions

Caption the image you *got*, not the prompt you *sent* — they diverge.

```
image ─▶ vision pass ─▶ scene JSON ─▶ voice pass ─▶ 4 candidates ─▶ score ─▶ pick
                                          ▲
                               voice card + last 50 captions
```

**Voice card = concrete rules, not adjectives.** "Warm and playful" produces
nothing usable. "always lowercase / max one emoji / no hashtags / fragments over
sentences / never explains the joke" produces a voice.

**Anti-repetition is what separates this from obvious bot output**, and it
needs two signals, not one. Bots read as bots less because they repeat *words*
than because they repeat *construction*: a feed where every caption is
"statement. two-word fragment." reads as automated even with entirely fresh
vocabulary.

- **Content** -- Jaccard over character trigrams, catching near-duplicate wording.
- **Shape** -- sentence count, coarse length buckets, whether it ends on a
  fragment. Bucketed rather than counted, because "bus is late. walking
  instead." and "coffee went cold. drinking it anyway." are the same rhythm.

Both are pure stdlib, so this costs nothing and needs no model. Eight captions
hand-written for the preview were run through it: the eighth was rejected as
the fourth use of one rhythm, which is the check doing its job on its author.

Candidates that break a machine-checkable voice rule (case, length, hashtags,
em dashes, emoji count) are rejected in code before scoring; the prose rules go
to the model. When every candidate fails, the captioner retries and finally
raises rather than posting something off-voice.

## (c) Conversation

```
inbound ─▶ policy gate ─▶ context assembly ─▶ model ─▶ outbound gate ─▶ send
              │                  │
          hard stops     history + facts + recent posts
```

Including *what she posted recently* is what makes replies cohere with the
timeline instead of floating free.

This runs in a local chat window (`python -m bot.ui`), which is how the persona
gets tuned: talk to her, watch which boundary fires, edit the voice card in the
side panel, talk again. Edits apply to the live prompt immediately and only
touch `persona.json` when you ask them to. Without an API key it still
classifies every message and shows the boundary, which is the half you can tune
for free.

**Gates are deterministic first, model second** — regex and keyword rules that
work regardless of which model is behind them, with a cheap classifier call as a
second layer. This is deliberate: it's the part that must not degrade when you
swap in a weaker free model.

| Trigger | Behavior |
|---|---|
| "are you a bot / real / AI?" | Answer truthfully. Always. No exceptions path. |
| Money, gifts, crypto, payment links | Hard block outbound. Never initiates, never accepts. |
| Address, workplace, financial details | Refuse in character. |
| Meetup / video-call framing | Deflect — the persona has no physical existence to offer. |
| Any signal the correspondent may be a minor | Terminate thread, flag for human review. The model is never called. |

## No platform integration

X integration is **dropped**. There is no `XDriver` and no posting to a live
service; the timeline exists locally so captions can be reviewed in context.

`SocialDriver` and `LocalDriver` stay as they are -- the storage they provide is
doing real work for the chat window and the timeline, and the protocol costs
nothing. If a platform is ever wanted again, that boundary is where it would go.

## Build order

1. Persona core + voice card ← everything blocks on this
2. Local storage + studio UI
3. Selfie pipeline. **3a** anchor + samples (browser, no GPU); **3b** bootstrap → LoRA → QC gate
4. Captioning with repetition scoring
5. Chat agent — boundaries written *before* generation logic
6. CLI joining the halves (`bot caption` fills the timeline)

All done except **3b**, which needs one GPU session: run
`notebooks/bootstrap.py` on Kaggle, then pin a LoRA trainer and script it.
