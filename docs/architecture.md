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
  checkpoint saved at the stop fits under it;
- resume: the data order is deterministic, so resuming at step *k* of an epoch replays exactly
  the rest of that epoch.

**Training backends** (`TrainingBackend`, `TrainingSession`, `Sampler`) own the provider's
side of a session:

- opening a session for a `ModelProfile` and `LoraConfig`, fresh or resumed from a training
  checkpoint, and releasing it when the context exits, however the run ended;
- converting each `TokenizedExample` to the SDK's own datum type;
- `forward_backward`, `forward` and `optim_step`, returning `PendingResult`s. A backend whose
  forward-only call returns per-token logprobs instead of a loss computes the loss itself, as
  the weighted sum of the negative logprobs;
- saving training checkpoints (weights and optimizer state, resumable) and sampler checkpoints
  (weights only), within the provider's checkpoint-name limit
  (`max_checkpoint_name_len`);
- promoting a sampler checkpoint to a deployable model, and sampling from the base model or a
  checkpoint.

**Renderers** (`Renderer`) own tokenization and response parsing. They are provider-neutral:
each model family has one renderer built on its Hugging Face chat template and tokenizer,
pinned to a revision, and every provider serving that family uses it.

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
credentials, validation of provider-specific `provider_options`, and `deploy` / `undeploy`.

## Training matches inference

The loop trains on one render per row with every assistant turn trainable. That costs
O(turns) tokens rather than O(turns²), but it is only correct when the renderer is
**prefix-stable**: for every assistant turn *i*, the generation prompt for `messages[:i]` must
be a token prefix of the full render, so the model is trained on exactly the tokens it will
see at inference.

Chat templates are often prefix-stable as text and still not as tokens, because a tokenizer can
merge characters across the point where generation starts. Renderers therefore tokenize
**segment-wise**: they split the rendered conversation at each generation-prompt end and each
assistant-turn end, and tokenize each segment on its own, which is how the inference server
tokenizes the prompt and the completion. A renderer that still cannot guarantee prefix
stability declares `prefix_stable = False` (or raises `RendererError` with code
`not_prefix_stable`), and the loop falls back to `per_turn_examples`.

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
to a `"module:attr"` import path. Listing providers imports nothing; `get_provider(name,
runtime)` imports the module on first use and calls `attr(runtime)`, which must return a
`Provider`. A provider whose SDK is not installed fails with a hint to install its extra.
Entries registered with `dev_only=True`, like the offline `fake` provider used for
development, are visible only with `PUP_FINETUNE_DEV=1`.

## How a new provider plugs in

Using Baseten as the example:

1. **Package.** Add `src/pup_finetune/providers/baseten/`. Only its `client.py` imports the
   provider SDK; every other module talks to that client, which keeps the SDK optional and
   the fakes simple.
2. **Backend.** Implement `TrainingBackend` with the Baseten Loops SDK. `open_session` returns
   a context manager that deactivates the session on exit, whether training finished,
   stopped early or failed. Convert each `TokenizedExample` to the SDK's datum type, map
   `AdamConfig` onto its optimizer parameters, and set `max_checkpoint_name_len` to the
   provider's limit.
3. **Renderer.** Reuse the Hugging Face chat-template renderer for the model family: the same
   model renders the same way on every provider. A new family needs a new renderer, plus a
   prefix-stability test against its real template.
4. **Profiles.** Return `ModelProfile`s from `models()`, with the tokenizer pinned to a
   40-character revision, and check in `preflight` that the provider's model matches the
   profile.
5. **Credentials.** Register a `CredentialSpec` for `BASETEN_API_KEY` in `credentials.py`'s
   registry, so `auth login baseten`, `auth status` and `doctor` work without new code. Pass
   the resolved key explicitly to the SDK; never copy it into the environment.
6. **Deploy.** Implement `deploy` and `undeploy` for a promoted `ModelRef`. Callers gate them
   behind `--confirm` and `PUP_READ_ONLY`.
7. **Registration and install.** Call `register_provider("baseten",
   "pup_finetune.providers.baseten:BasetenProvider")` in the registry, and add a `baseten`
   extra to `pyproject.toml` for the SDK (with its rows in `LICENSE-3rdparty.csv`).
8. **Tests.** Subclass the contract suites in `tests/providers/contract/`
   (`RendererContract`, `BackendContract`, `ProviderContract`) against a fake of the SDK
   client, so the default suite makes no network or paid calls, and gate live tests behind
   `PUP_FINETUNE_LIVE=1`. [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-provider) has the
   full checklist.

The shared loop, resume, evaluation and the CLI then work for the new provider unchanged.
