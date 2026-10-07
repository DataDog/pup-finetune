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
  an OpenAI assistant message plus a `format_valid` flag. `parse_response` takes the prompt's
  tools, since a format can write argument values untyped (Qwen3.8 writes strings unquoted).
- `identity()` pins the tokenizer, its revision, the template hash and the renderer and parser
  versions. A run records it, and a mismatch on resume means "prepare a new run".

**Providers** (`Provider`) own everything outside the loop: the model profiles they support
(`models`, `resolve_model`), a live `preflight`, readiness `check`s for `doctor`, their
credentials, validation of provider-specific `provider_options`, `training_profile` (the
profile as those options train it, e.g. with a shorter context: rows are rendered against
it and the run records it, so a preview shows its effect and a resume renders the same rows),
and `deploy` / `smoke_test` / `undeploy`.

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
per turn, as does `qwen3_5` (Qwen3.5 also drops reasoning before the last user turn); every
turn boundary of GLM-5.3 is a special token, so `glm5_3` takes the fast path. Only
`supervised_example` raises `not_prefix_stable`; a per-turn failure carries its own code, so
the loop never retries it per turn.

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
[data-format.md](data-format.md) has the full format, an example and the exclusion codes.

## The registry

[`providers/__init__.py`](../src/pup_finetune/providers/__init__.py) maps each provider name
to a `"module:attr"` import path. Listing providers imports no provider implementation;
`get_provider(name, runtime)` imports the module on first use and calls `attr(runtime)`, which
must return a `Provider`. A provider whose SDK is not installed fails with a hint to install its extra.
Entries registered with `dev_only=True`, like the offline `fake` provider used for
development, are visible only with `PUP_FINETUNE_DEV=1`. `fireworks` and `baseten` are
registered for everyone. The registry also imports the lightweight package of every provider
that has an API key (today `fireworks` and `baseten`, through imports at the bottom of
`providers/__init__.py`). That
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

## Training on Baseten

[`providers/baseten/backend.py`](../src/pup_finetune/providers/baseten/backend.py) trains on
Baseten Loops through `providers/baseten/client.py`, the only module that loads the Loops SDK:

- **Preflight.** Read only: the SDK loads, and Loops' capabilities must offer and enable the
  model to the workspace with a limit of at least the `max_seq_len` option, else the
  profile's context. Nothing is paid before `--confirm`.
- **Sessions.** `open_session` opens a Loops session and a dedicated LoRA trainer whose
  context is the next power of two over the session's longest example (at least 1024 tokens,
  at most the profile's). Leaving the context always deactivates the run and every sampler
  the session started; a deactivation that fails names what to deactivate by hand. With no
  capacity, nothing is left running.
- **Checkpoints.** `bt://loops:` paths that outlive the run; a resume loads a training
  checkpoint with its optimizer state into a new run. There is no promotion: the best sampler
  checkpoint is the run's deployable checkpoint.
- **Sampling.** Each sampler runs on its own dedicated deployment, sized to the profile's
  context, and renders, samples and parses as on Fireworks. The `max_seq_len` provider option
  cuts the training profile, so it also sizes the samplers and the trainer a standalone
  evaluation opens.

## Replay evaluation

[`evaluation/`](../src/pup_finetune/evaluation/) is provider-neutral: it never imports a
provider, and talks to models only through the `Sampler` protocol.

- **Cases** ([`cases.py`](../src/pup_finetune/evaluation/cases.py)). Every assistant message
  of a kept test-split row that does something visible (non-empty content or tool calls) and
  was trained on (`weight` is not `0`) is a `ReplayCase`, with id `<row id>:<message index>`.
  The model sees the recorded messages before it plus the row's tools, generates one
  assistant message, and that message is compared with the recorded one, the reference. The
  tool results that followed the reference are kept for a judge only. Tools are never
  executed, so the conversation continues from the recording: replay measures agreement with
  the recorded behavior, not whether the task succeeded. A per-trajectory cap keeps evenly
  spaced turns, one per equal bucket at a seeded offset, and always the final turn.
- **Metrics** ([`metrics.py`](../src/pup_finetune/evaluation/metrics.py)). Pure functions of
  the candidate, the reference and the tools, each `true`, `false`, or `null` when it does not
  apply: `tool_decision_match`, `tool_name_match`, `arguments_json_valid`,
  `arguments_schema_valid` (against the declared `parameters`, with a small validator for the
  common JSON Schema subset; other keywords are ignored), `reference_arguments_match`,
  `parallel_call_set_match` (both order-insensitive for parallel calls) and `format_valid`,
  from the sampler. Arguments compare as JSON values, and `reasoning_content` is never
  compared. `aggregate` reports `{rate, passed, n}` per metric over the cases it applies to,
  with `rate` `null` when `n` is `0`.
- **Judge** ([`judge.py`](../src/pup_finetune/evaluation/judge.py)). An LLM on any judge
  provider (see [Judges](#judges)) says whether the candidate is an acceptable substitute for
  the reference. The recorded data is untrusted and travels JSON-encoded in tagged blocks.
  The answer is parsed strictly whichever provider gave it; an answer the provider refused
  (`finish_reason` `refusal`) is a failed judgment with error `refused`, like an unparseable
  one.
  Before anything is generated, a calibration checks that references pass and corrupted
  versions of them fail; a judge that fails it is not used. The judge must not be the model
  being trained, nor the run's tuned model (its model id, or the promoted model).
- **Targets** ([`targets.py`](../src/pup_finetune/evaluation/targets.py)). Two samplers from
  one session: `base` (no checkpoint) and `tuned`, the best epoch. `train --evaluate` runs in
  the training session through the runner's `after_finalize` hook, after promotion and
  before the session closes, and samples the best epoch's own sampler checkpoint. The runner
  records the run's end (`result.json`, phase `completed`) before the hook runs, so a process
  killed during the evaluation leaves a finished run, not one a resume would train again. A
  cancelled or budget-capped training is not evaluated in its session, since the evaluation
  would spend the session's time past the cap. Any other is evaluated under the cap too: the
  hook gets the run's spend guard (its reserve included), skips the evaluation when the cap
  is already reached and stops it, partial, once it is, for `evaluate` to continue.
  A standalone `evaluate`, of a completed or cancelled run with a best epoch, opens a new
  session, loads the best *training* checkpoint and saves it again as a sampler checkpoint,
  since a sampler checkpoint may not outlive the session that saved it.
- **Replay** ([`replay.py`](../src/pup_finetune/evaluation/replay.py)). `run_replay` runs
  every `(target, case)` pair on a thread pool: generate, score, judge. Each generation and
  each result is appended to `generations.jsonl` and `results.jsonl` as soon as it is known,
  so an interrupted evaluation loses only the pairs in flight, and a rerun skips the pairs
  already judged. `evaluation.json` pins what the results depend on (the run, the tuned
  weights, the base model, the judge, the cases, `max_tokens`, the rendering, the metrics
  version, the seed, the cap); a rerun with anything else is refused and asks for a new
  `--out-dir`. A generation or judge call that fails after its client's retries is an error
  result: it is retried by a rerun, counted on its own, and never counts as a failed case;
  an evaluation that ended with errors reports `complete_with_errors`, not `complete`, and
  how to retry them. A judge answer with no verdict (unparseable or truncated) is a failed
  case, as the judge decides, and the summary counts those per target
  (`judge_no_verdict`). Ctrl-C or SIGTERM stops new pairs from starting and keeps what
  finished. The directory is made `0700` even when it already existed.
- **Summary** (`summary.json`). Per target, the judge pass rate, every metric's rate and the
  mean of per-trajectory agreement. Across targets, the cases both answered are compared
  verdict by verdict (wins, regressions, ties), and `tuned_minus_base` is the mean paired
  difference of the judge pass, with a 95% bootstrap interval that resamples whole
  trajectories (2000 seeded resamples; `null` with fewer than two trajectories), since the
  cases of one conversation are not independent. Every number is finite or `null`, and the
  summary says what the scores measure: agreement with the recorded behavior on held-out
  conversations, not task success.
- **Runner** ([`runner.py`](../src/pup_finetune/evaluation/runner.py)). What `evaluate` and
  `train --evaluate` run, so the CLI only parses flags and emits: an `EvalPlan` (cases,
  judge, settings, and whether the evaluation can be published), calibration, the replay
  with the plan's publisher, and the result and exit code. The package exports its names
  lazily, so the CLI loads the runner for every command without loading the judge's client.
- **Publishing** ([`datadog/experiments.py`](../src/pup_finetune/datadog/experiments.py)).
  `Publisher` receives a complete evaluation and returns what it reports. Datadog is the
  system of record: `ExperimentsPublisher` publishes the evaluation, through pup's OAuth, as
  one Agent Observability experiment per target on the curated dataset `dataset push`
  created, pinned to the pushed version (`push.json`), so the training data and the
  evaluation of the model trained on it stay on one dataset and the two experiments compare
  side by side. Its config holds the provider, the models, the tuned checkpoint, the run, the
  judge and the metrics version. Each test row is a root span, `replay_trajectory`, linked to
  its dataset record and holding the record's input and expected output, with the replayed
  steps as its output; each case is a child span in the same trace, `replay_assistant`, with
  the prefix and tools in, the generated message out and the recorded one expected, timed by
  its generation. Metrics join spans by id: a boolean per deterministic metric that applies
  and `judge_pass` (with the judge's reason) on children, `trajectory_agreement` and
  `completed_actions` on roots; an error case is an error span with no metric, as it counts
  in no local rate either. Metrics are stamped when they are sent (the API refuses ones more
  than a day old), and a request carries at most 100 spans. Span ids are uuid5 of the
  evaluation's identity, the target and the row or case, the experiments are recorded in
  `experiments.json` and the spans sent in `published.json`, so a rerun creates and sends
  nothing twice. Each create is recorded as pending first, with a random nonce that goes in
  the experiment's description, so a rerun after a crash between the create and the save
  finds that experiment and adopts it (another `--out-dir` has its own nonce, and never
  does). A span is recorded once its request is answered, so a batch whose
  response was lost is sent again by the rerun, with the same deterministic span ids. A
  span already published keeps the result it was sent with: a retried error updates only
  the local results, and the experiments keep the error (the result's hint says so, and a
  new `--out-dir` publishes experiments with every result). A cancel stops between requests
  and leaves a `cancelled` publication the rerun continues. A read-back waits up to two
  minutes, one budget for both experiments, for the events to be indexed (each count may
  take only the time left, and a cancel is checked before each); spans not indexed
  by then, or events that cannot be counted, are a warning. Without a push receipt the evaluation is not
  published, with a warning; `--no-publish` turns publishing off.

The fake provider makes the loop meaningful offline: its tuned checkpoints learn, from the
conversations they trained on, whether an assistant turn after a message of each role calls
a tool (and which declared tool, by position) or answers in text, and `FakeJudge` is an
offline judge that passes the same tool calls with consistent arguments, or text with mostly
the same words. So a development-mode `evaluate` of a fake run shows the tuned model ahead
of the base model.

## Deploying

[`deployment.py`](../src/pup_finetune/deployment.py) is what `deploy` and `undeploy` run, and
it is provider-neutral: it deploys a finished run's promoted model (or, on a provider without
promotion, its deployable checkpoint) through `Provider.deploy`, with `DeployOptions` (the
deployment id, derived from the run id unless `--deployment-id` is given; the shape; at most
one replica, none when idle; the wait's timeout). The receipt, `deployment.json` in the run
directory, is first written as `REQUESTED` from the provider's `on_creating` callback, right
before the create is sent and after any lookup it needs (the shape), so a failure before
that leaves no receipt and a Ctrl-C while the deployment is created still leaves its id to
undeploy, with the resource name it gets (`Provider.deployment_name`, on Fireworks
`accounts/<account>/deployments/<id>`), so an undeploy under another account is refused, and
what the create sends (`DeploymentRequest`: the base model, display name and shape), as its
creation is never confirmed: `undeploy` reads the deployment under the id back
(`Provider.owns_deployment`) and deletes it only when it is the one requested, takes it as
gone when there is none, and refuses when the id holds another deployment (the receipt is
removed again when the provider refused the request). The `on_created` callback rewrites it as soon as the provider
accepted the deployment and before the wait, so a failure, a timeout or a Ctrl-C while
waiting still records what exists and bills; a smoke test that raises is recorded as failed,
with an undeploy hint. It is updated once the deployment serves, with the
`smoke_test` result, and with `deleted_at` by `undeploy`. A run has one deployment at a time.
`undeploy` is idempotent and keeps the model. Both are remote writes: they preview without
`--confirm` and are refused with `PUP_READ_ONLY`.

On Fireworks, [`providers/fireworks/deploy.py`](../src/pup_finetune/providers/fireworks/deploy.py)
drives the SDK's `Fireworks` client through `client.py`: the first validated, latest-validated
shape `deployment_shape_versions.match_for_model` returns (unless `--shape`), then
`deployments.create` (sent once, without the SDK's retries: the API takes no idempotency key,
so a retried create that had succeeded would answer HTTP 409; after a 409, a timeout or a
server error the deployment is read back and adopted only when it is this deploy's own:
the requested model, display name, shape and replica counts; any other is refused as a
taken `--deployment-id`, so it is never recorded, or undeployed, as the run's)
with the promoted model as the base model, `min_replica_count` 0,
`max_replica_count` 1, a five-minute scale-to-zero window and hot loading off (the SDK's own
deployment helper defaults it on, for training rollouts). It reads the deployment every 15
seconds until `READY` (`FAILED`, `DELETING` or `DELETED` stop at once, as
`DeploymentNotReady`), then smoke-tests one chat completion through
`https://api.fireworks.ai/inference/v1` naming the deployment
(`accounts/<account>/deployments/<id>`), retrying HTTP 503, 429 and dropped connections
while a replica starts from zero. `undeploy` refuses a deployment whose resource name places it
in another account than the key resolves to, where a 404 would not mean it is gone.
Fireworks' own CLI refuses to create deployments when an AI agent runs it; pup-finetune uses
the API and relies on `--confirm`, so an agent confirms only with the user's explicit
authorization.

## Judges

LLM judges (the replay judge and the triage council) run on Fireworks, Anthropic or OpenAI.
[`judges.py`](../src/pup_finetune/judges.py) is provider-neutral and imports no SDK:
`parse_judge` reads a judge spec by prefix (`anthropic/<model>` or `openai/<model>`; any
other spec is a Fireworks model id such as `accounts/fireworks/models/<model>`, as before)
into a `JudgeConfig` with the provider's defaults (endpoint, credential,
whether a temperature is sent, the token limit). `JudgeConfig.provider` is part of the
judge's identity, so a judge on another provider never reuses earlier judgments or votes, and
`JudgeConfig.spec` writes the judge back the one way commands pin it. `open_client` builds
the client for a judge, and `open_clients` one client per provider for a council that mixes
them; every client answers the one `ChatCompleter.complete` call, which takes an optional
`response_schema` for a JSON answer.

- **Fireworks** and **OpenAI** share the `httpx` `ChatClient`
  ([`inference.py`](../src/pup_finetune/inference.py)), with a per-provider `RequestPolicy`:
  OpenAI's token limit goes in `max_completion_tokens`, no `temperature` is sent (its current
  models reason), a JSON answer is asked for with `response_format` `json_object`, an
  answer with a `refusal` (or `finish_reason` `content_filter`) is reported as a refusal, and
  each attempt may take up to 10 minutes, as a reasoning model's answer arrives only once it
  is complete.
- **Anthropic** uses the official SDK (the `anthropic` extra) in
  [`anthropic_client.py`](../src/pup_finetune/anthropic_client.py), the only module that
  imports it: the leading system message becomes `system`, no sampling parameter is sent,
  `output_config.format` carries the verdict's JSON schema on models with structured outputs,
  `output_config.effort` is `low` on models with effort levels, and the models with
  server-side refusal fallbacks are called through the beta API with `fallbacks="default"`.
  Its typed errors map to the same error classes as `ChatClient`'s; a failed SDK sign-in
  (`CredentialsError`, `WorkloadIdentityError`) is a rejected key, which stops a triage.

The keys are `CredentialSpec`s like Fireworks' ([`judge_credentials.py`](../src/pup_finetune/judge_credentials.py),
registered when the providers package is imported): `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`,
or the keychain accounts `anthropic` and `openai`. Without an Anthropic key, the SDK may sign in
on its own (`ANTHROPIC_AUTH_TOKEN`, an `ant auth login` profile whose file exists); `auth
status` and `doctor` report that as the source `anthropic-sdk`, and the tokens it would use
are registered for redaction before the client is built. Every preview names each provider and endpoint the
data is sent to.

## Triage

[`curation/triage.py`](../src/pup_finetune/curation/triage.py) is `dataset triage`, the paid
alternative to agent or human labels: a council of judge models (two Fireworks serverless
models by default; any [judge provider](#judges), mixed as the user likes) votes keep
or drop on each **whole** row against a rubric or rules, since training imitates every
assistant turn of a kept row. The prompt follows the replay judge's conventions: the system
message holds the rubric and says that everything recorded is untrusted data, and the row
travels JSON-encoded in tagged blocks it cannot close. The answer is parsed strictly as
`{"keep": bool, "reason": str}`.

A row is kept only on a strict majority of keep votes; a tie drops it; a row missing a vote
is undecided and left unlabeled. A failed call, an answer without a vote, or one the
provider refused, is not a vote:
it is reported and retried by the next run. Votes are appended to `judgments.jsonl` as they
arrive, and count only for the row content, rubric and judge they were cast for, so an
interrupted triage resumes. The rubric's hash and the judges (as provider-qualified specs)
are frozen in `triage.json`
once votes exist (`--reset-triage` discards them). Decisions are recorded as labels with
`by: judge` through `apply_labels`; rows the agent or a person labeled are not reviewed. A
label cannot be withdrawn, so a judge label the current votes do not back (cast under a
discarded rubric, outside a `--max-rows` sample, or on a row left undecided) is reported as
stale, and the triage is `incomplete` until a run over every row replaces it.
`--max-rows` reviews a fixed pseudo-random sample, so a calibration run's votes are reused
by the full run. A rubric template ships with the package (`--print-template`).

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
6. **Deploy.** Implement `deploy` (call `on_creating` with what the create sends right
   before sending it, `on_created` once the deployment exists, then wait
   until it serves, raising `DeploymentNotReady` when it fails or the wait runs out),
   `smoke_test` (one short chat completion, retried while it scales up; a failed answer is a
   result, not an exception), `owns_deployment` (whether the deployment under an id is the
   one a `DeploymentRequest` creates) and an idempotent `undeploy` for a promoted `ModelRef`. Callers
   gate them behind `--confirm` and `PUP_READ_ONLY`.
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
