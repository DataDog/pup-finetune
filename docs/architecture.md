# Architecture: the training seam

This page is for contributors. It explains how `pup-finetune` splits fine-tuning between
provider-neutral code and the per-provider pieces, so that a new training provider adds a
small package instead of new branches in the CLI.

The protocols live in [`src/pup_finetune/providers/base.py`](../src/pup_finetune/providers/base.py)
and [`src/pup_finetune/rendering/base.py`](../src/pup_finetune/rendering/base.py). The
development-only `fake` provider in
[`src/pup_finetune/providers/fake.py`](../src/pup_finetune/providers/fake.py) implements all of
them offline and is the reference implementation. The shared training loop is
[`src/pup_finetune/training/loop.py`](../src/pup_finetune/training/loop.py).

## Who owns what

```
 TrainingRow (data/schema.py)
      |
      v
 Renderer ---------> TokenizedExample ----------> TrainingBackend / TrainingSession
 (tokenize, parse)   (provider-neutral arrays)    (sessions, checkpoints, promotion, sampling)
      ^                                                 ^
      |                                                 |
 Provider: model profiles, preflight, credentials, deploy
      ^
      |
 training loop: epochs, batching, eval, early stopping, checkpoint policy, resume
```

**The shared loop** (`training/`) is written once, against the protocols only.
It owns everything that decides *how* training goes:

- epochs, per-epoch shuffling of whole rows seeded with `seed + epoch - 1`, and batching
  (`batch_size` rows per step, packed into microbatches of at most `microbatch_token_budget`
  tokens, up to `pipeline_depth` steps in flight, every pending result always resolved);
- never retrying `forward_backward`, since a retry would accumulate the gradients twice;
- validation loss after each epoch through `TrainingSession.forward`, retried with a short
  backoff a bounded number of times when it fails with a `TransientError` (or a timeout or
  connection error);
- early stopping by `patience` and `min_delta`, and tracking the best epoch;
- the checkpoint policy: a resumable checkpoint at each epoch end and every
  `checkpoint_every_minutes` at a step boundary, and a sampler checkpoint for each new best;
- the `max_train_tokens` budget and the `max_spend_usd` / `hourly_rate_usd` wall-clock spend
  guard, which stop cleanly and still promote the best checkpoint; the spend guard is asked
  before every paid call and stops training `spend_reserve_fraction` short of the cap, so the
  checkpoint saved at the stop and the promotion fit under it, and the time spent carries
  over a resume;
- resume: the data order is deterministic, so resuming at step *k* of an epoch replays exactly
  the rest of that epoch. Each training checkpoint records its epoch, step, the steps already
  trained in the epoch and the epoch's loss so far, so a resumed run ends with the same epoch
  metrics, best epoch and patience counter as an uninterrupted one;
- promote-before-exit: however training stops, the best sampler checkpoint is promoted while
  the session is still open. After a resume it may come from an ended session, so it is saved
  again from the best training checkpoint first. Without promotion, the best sampler
  checkpoint is recorded as the deployable artifact. A failed promotion keeps the training
  result and leaves the run resumable for the promotion alone.

**Training backends** (`TrainingBackend`, `TrainingSession`, `Sampler`) own the provider's
side of a session:

- opening a session for a `ModelProfile` and `LoraConfig` (sized by the run's `seed` and
  `max_seq_len`, the longest example it trains on, at most the profile's context limit),
  fresh or resumed from a training checkpoint, and releasing it when the context exits,
  however the run ended;
- declaring the optional capabilities `supports_resume` and `supports_promotion`, which the
  loop checks before relying on them;
- converting each `TokenizedExample` to the SDK's own datum type;
- `forward_backward`, `forward` and `optim_step`, returning `PendingResult`s. A backend whose
  forward-only call returns per-token logprobs instead of a loss computes the loss itself, as
  the weighted sum of the negative logprobs;
- saving training checkpoints (weights and optimizer state, resumable) and sampler checkpoints
  (weights only), within the provider's checkpoint-name limit
  (`max_checkpoint_name_len`);
- promoting a sampler checkpoint to a deployable model (where supported), loading a
  checkpoint's weights, and sampling from the base model or a checkpoint.

**Renderers** (`Renderer`) own tokenization and response parsing. They are provider-neutral:
one implementation, `HFChatTemplateRenderer` in
[`rendering/hf.py`](../src/pup_finetune/rendering/hf.py), renders with the model's own
Hugging Face chat template and tokenizer, pinned to a revision, and every provider serving
that model uses it. Each model family is an entry in
[`rendering/families.py`](../src/pup_finetune/rendering/families.py): the tokenizer, its
40-character revision, the response parser (from
[`rendering/parsers.py`](../src/pup_finetune/rendering/parsers.py)), the stop tokens, any
template arguments, and for models that stop by emitting the next role tag, which tag follows
an assistant message. `transformers` comes from the optional `hf` extra and loads lazily.

- `supervised_example` renders a whole conversation with every assistant turn trainable
  (honoring `weight: 0`); `per_turn_examples` renders each trainable turn against its own
  prefix.
- `generation_prompt_tokens`, `stop_token_ids`, `decode` and `parse_response` are the
  inference side: the prompt a sampler continues, where it stops, and how a completion becomes
  an OpenAI assistant message plus a `format_valid` flag.
- `identity()` pins the tokenizer, its revision, the template hash and the renderer and parser
  versions. A run records it, and a mismatch on resume means "prepare a new run".

**Providers** (`Provider`) own everything outside the loop: the model profiles they support
(`models`, `resolve_model`), a live `preflight`, readiness `check`s for `doctor`, their
credentials, validation of provider-specific `provider_options`, `training_profile` (the
profile as those options train it, e.g. with a shorter context: rows are rendered against
it and the run records it, so a preview shows its effect and a resume renders the same rows),
and `deploy` / `undeploy`.

## Training matches inference

A chat endpoint renders the conversation with the chat template, tokenizes the whole prompt
text as **one** string, and the model generates the completion after it. Training must see the
same tokens, so by default `HFChatTemplateRenderer` renders **one example per trainable
assistant turn**: the generation prompt for `messages[:i]`, tokenized whole, followed by the
turn's completion tokens. The loss covers the completion through its stop token and nothing
after it (for role-tag families, the appended next role tag is the stop target, after any
whitespace the template renders past the reply is dropped). A turn whose render does not
continue its own prompt is rejected with `prompt_mismatch`.

Chat templates are often prefix-stable as text and still not as tokens: a tokenizer can merge
characters across the point where generation starts. Qwen3.8's prompt ends `<think>\n` and an
empty reasoning block continues `\n`, which the whole conversation encodes as one `\n\n`
token. In that case the completion is the rest of the rendered turn encoded on its own after
the prompt's tokens, as the model would generate it.

One example per row with every assistant turn trainable costs O(turns) tokens rather than
O(turns²), but it is only correct when the renderer is **prefix-stable** for that row: for
every trainable turn *i*, the turn's per-turn example must be an exact token prefix of the
whole conversation tokenized once. A family may pin `prefix_stable` to try this fast path;
the renderer then checks it on every row in `supervised_example` (rows with
`reasoning_content` always render per turn) and raises `RendererError` with code
`not_prefix_stable` when it does not hold, and the loop renders that row with
`per_turn_examples`. A renderer that is not prefix-stable declares `prefix_stable = False` and
the loop renders every row per turn. Qwen3.8 is not token prefix-stable, so `qwen3_8` renders
per turn. Only `supervised_example` raises `not_prefix_stable`; a per-turn failure carries
its own code, so the loop never retries it per turn.

A tokenizer that cannot be loaded fails every row alike, so the renderer reports it as an
`ExternalError` (a `UsageError` when the `hf` extra is missing) rather than rejecting rows.

The Qwen3.8 response parser knows its generation prompt opens `<think>`: sampled text that
never reaches `</think>` (cut off at the token limit) is all reasoning and `format_valid` is
false.

## Token arrays

A renderer emits a `TokenizedExample` (`rendering/tokens.py`), shifted for next-token
prediction:

- `input_tokens = tokens[:-1]`, `target_tokens = tokens[1:]`;
- `weights[i]` is the loss weight of `target_tokens[i]`: `1.0` on assistant-completion tokens,
  `0.0` on everything else;
- `TokenizedExample.from_tokens(tokens, token_weights)` does the shift: the weight of token
  `i + 1` attaches to target `i`.

Backends never see chat messages during training, only these arrays. The loss is reported
as `StepMetrics(loss_sum, weight, tokens)`, and `mean_loss` is `loss_sum / weight`: the
per-token mean over the batch.

## The data contract

Every provider trains on the canonical row in
[`data/schema.py`](../src/pup_finetune/data/schema.py), version `1`:

- `id`, `messages`, `tools`, `source` and `schema_version`.
- `messages` are in OpenAI chat format with roles `system`, `user`, `assistant` and `tool`.
  A system message may only come first, and the last message is from the assistant.
- Assistant messages may carry `tool_calls` (`arguments` is a JSON object encoded as a
  string), `reasoning_content`, and `weight` (`0` means "do not train on this turn").
- A `tool` message answers an earlier call through `tool_call_id`. Called tools must be
  declared in `tools` when any are declared.
- `source` records where the row came from (trace and span ids, `ml_app`, model, tags).

Renderers receive rows in the OpenAI form that
[`data/jsonl.py`](../src/pup_finetune/data/jsonl.py)'s `to_provider_dict` produces, and adapt
it to their template (for example, decoding `arguments` for templates that expect an object).
Rows are validated once, when they enter a dataset, so renderers can rely on this shape.

## The registry

[`providers/__init__.py`](../src/pup_finetune/providers/__init__.py) maps each provider name
to a `"module:attr"` import path. Listing providers imports no provider implementation;
`get_provider(name, runtime)` imports the module on first use and calls `attr(runtime)`, which
must return a `Provider`. A provider whose SDK is not installed fails with a hint to install its extra.
Entries registered with `dev_only=True`, like the offline `fake` provider used for
development, are visible only with `PUP_FINETUNE_DEV=1`. `fireworks` is registered for
everyone. The registry also imports the lightweight package of every provider that has an API
key (today `fireworks`, through an import at the bottom of `providers/__init__.py`). That
package's `__init__.py` does little more than register the provider's `CredentialSpec`, so
`auth`, `doctor` and every other command know the API key without importing an SDK, an HTTP
client or a tokenizer. The implementation lives in the package's `provider` module, which
`get_provider` imports on first use.

## Training on Fireworks

[`providers/fireworks/backend.py`](../src/pup_finetune/providers/fireworks/backend.py) is the
`TrainingBackend` for Fireworks serverless training, and the worked example of a real one. It
talks to the narrow interface in `providers/fireworks/client.py`, never to the SDK:

- **Sessions.** `open_session` opens one serverless training session and creates the LoRA
  run on it (rank, alpha and seed from the run), or continues a training checkpoint with
  `create_training_client_from_state_with_optimizer`. Leaving the context always ends the
  session: results still in flight are waited for briefly after a normal exit and abandoned
  after an error or Ctrl-C, then the samplers, the session and the client are closed. A
  failure to close never hides the error that ended the body, and a failure to close the
  session (which may leave it billing) is the one raised after a normal exit, any other as a
  note. With no capacity, nothing is left running. The `max_seq_len` provider option cuts the
  profile's context (`training_profile`), so longer rows are left out when the rows are
  rendered; the session also refuses a longer sequence length, as a backstop.
- **Losses.** `forward_backward` reports Fireworks' `loss:sum` over `response_tokens`.
  Fireworks' forward pass returns no loss, only per-token logprobs, so the validation loss is
  `-sum(logprob * weight)` over every target token, after checking that the logprobs match
  the examples sent, token for token.
- **Checkpoints.** Training checkpoints (`save_state`) hold weights and optimizer state and
  can be resumed in a later session; sampler checkpoints (`save_weights_for_sampler`) can be
  sampled from and promoted only while their session lives, which ends shortly after the
  process exits. Fireworks enforces its 17-character name limit only at promotion, so names
  are checked before saving.
- **Promotion.** Fireworks renames checkpoints (`run-<hex>-<name>-<8 hex>`), so `promote`
  lists the session's checkpoints and takes the newest promotable sampler row carrying the
  checkpoint's name, then promotes it to `accounts/<account>/models/<model id>`: the
  `ModelRef`'s `model_id` is the bare id, its `uri` that full name. The shared
  `finalize` hook ([`training/promotion.py`](../src/pup_finetune/training/promotion.py)) runs
  it before the session ends; after a resume, it first re-creates the best epoch's sampler
  checkpoint from its training checkpoint (`load_state`, then `save_weights_for_sampler`).
- **Sampling.** A sampler renders the prompt with the profile's renderer, samples greedily up
  to the renderer's stop tokens and parses the completion into an OpenAI message; a
  completion cut off at `max_tokens` is not `format_valid`. Sampling alone is retried on a
  transient failure, a bounded number of times with a warning before each retry; training
  calls never are, since the loop owns that policy. `latency_s` times the successful attempt
  only.

## How a new provider plugs in

Using Baseten as the example:

1. **Package.** Add `src/pup_finetune/providers/baseten/`. Only its `client.py` imports the
   provider SDK; every other module talks to that client, which keeps the SDK optional and
   the fakes simple.
2. **Backend.** Implement `TrainingBackend` with the Baseten Loops SDK. `open_session` returns
   a context manager that deactivates the session on exit, whether training finished,
   stopped early or failed. Convert each `TokenizedExample` to the SDK's datum type, map
   `AdamConfig` onto its optimizer parameters, size the trainer from the `seed` and
   `max_seq_len` passed to `open_session`, set `max_checkpoint_name_len` to the provider's
   limit, and declare `supports_resume` and `supports_promotion`. Map SDK failures onto the
   neutral errors in `providers/base.py`: `ProviderCapacityError` when there is no capacity
   or quota (end what the failed call started before raising it, and say so in the hint
   only then), `ProviderAuthError` for a rejected key and
   `TransientError` for a retryable server error or dropped connection, with redacted
   messages (`providers/fireworks/client.py` does this in `fireworks_error`).
   `providers/fireworks/backend.py` is a complete backend to follow.
3. **Renderer.** Reuse `HFChatTemplateRenderer` through the model's family in
   `rendering/families.py`: the same model renders the same way on every provider. A new
   family is a new table entry (plus a response parser when its output format is new) and
   `tokenizer`-marked tests against its real template, including the `RendererContract`.
4. **Profiles.** Return `ModelProfile`s from `models()`, with the tokenizer pinned to a
   40-character revision, and check in `preflight` that the provider's model matches the
   profile.
5. **Credentials.** Register a `CredentialSpec` for `BASETEN_API_KEY` in `credentials.py`'s
   registry from the package's lightweight `__init__.py` (as `providers/fireworks/` does), and
   import that package at the bottom of `providers/__init__.py` with
   `from pup_finetune.providers import baseten as _baseten  # noqa: E402, F401`, so
   `auth login baseten`, `auth status` and `doctor` work without new code. Pass the resolved
   key explicitly to the SDK; never copy it into the environment.
6. **Deploy.** Implement `deploy` and `undeploy` for a promoted `ModelRef`. Callers gate them
   behind `--confirm` and `PUP_READ_ONLY`.
7. **Registration and install.** Call `register_provider("baseten",
   "pup_finetune.providers.baseten.provider:BasetenProvider")` in the registry (the
   implementation lives in `provider.py`, so the package `__init__.py` stays light), and add a `baseten`
   extra to `pyproject.toml` for the SDK (with its rows in `LICENSE-3rdparty.csv`).
8. **Tests.** Subclass the contract suites in `tests/providers/contract/`
   (`RendererContract`, `BackendContract`, `ProviderContract`) against a fake of the SDK
   client, so the default suite makes no network or paid calls, and gate live tests behind
   `PUP_FINETUNE_LIVE=1`. [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-provider) has the
   full checklist.

The shared loop, resume, evaluation and the CLI then work for the new provider unchanged.
