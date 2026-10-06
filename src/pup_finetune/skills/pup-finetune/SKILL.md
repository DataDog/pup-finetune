---
name: pup-finetune
description: Fine-tune an open-weight model (Qwen3.5-9B or GLM-5.3-Flash) with LoRA SFT on Baseten Loops from canonical training-row JSONL files with pup finetune train, serve the best checkpoint behind an OpenAI-compatible endpoint with pup finetune deploy, and stop it with pup finetune undeploy. Use when a user wants to set up pup finetune for Baseten, train on their own train and validation JSONL files, deploy or undeploy a tuned model, or find out why a pup finetune train, deploy or undeploy failed.
---

# pup-finetune on Baseten

`pup finetune` fine-tunes a model on Baseten Loops from JSONL files and serves the result:

```text
doctor -> train (preview) -> train --confirm -> deploy (preview) -> deploy --confirm -> undeploy --confirm
```

Datasets from Datadog traces (pull, triage, push, prepare), evaluation and experiments arrive
in later releases; today the input is JSONL files the user already has. Run
`pup finetune <command> --help` for every option. `pup-finetune` is the same CLI without pup.

## Rules for every step

- **Read the JSON.** Each command prints one JSON result on stdout (pup may wrap it in its
  agent envelope), or none when it fails, and progress, `warning:`, `Error:` and `Hint:` lines
  on stderr. Exit codes: `0` success, `1` external or runtime failure, `2` usage error or a
  preview. When the result has a `next_command`, run that instead of building the next command
  yourself.
- **Preview, then ask.** `train`, `deploy` and `undeploy` do nothing paid or remote without
  `--confirm`. Without it they print `"status": "confirmation_required"`, the `preview`, and a
  `next_command` that ends in `--confirm`, and exit 2. Show the preview to the user. Run
  `next_command` only after the user explicitly authorizes that operation and its spend. Never
  add `--confirm` on your own; `PUP_AUTO_APPROVE` and agent mode do not replace it. One
  authorization covers one command; a changed flag needs a new preview.
- **Honor `PUP_READ_ONLY`.** When it is `true`, `train`, `deploy` and `undeploy` refuse, even
  to preview. Tell the user; do not unset it yourself.
- **Keep the API key secret.** Never print, echo or log it, never pass it as a flag, and never
  ask the user to paste it into the chat. The user stores it with
  `pup finetune auth login baseten` in their own terminal, or sets `BASETEN_API_KEY`.
- **Keep run directories.** Never delete, clear or edit files in a run directory (the one
  exception is in Debugging): they record what was trained, what is deployed and what must be
  stopped. Use a new run directory per run.
- **Do not guess.** Ask the user for the data files, the model, and any spend cap or price.

## 0. Setup

```sh
pip install 'pup-finetune[baseten]'     # the Loops SDK and the tokenizer libraries
pup finetune --version
pup finetune auth status baseten        # configured, and source: env or keychain
pup finetune doctor
```

If `auth status` shows `"configured": false` (or `doctor`'s `credentials:baseten` fails), ask
the user to run this in their own terminal (it prompts with hidden input), or to set
`BASETEN_API_KEY`:

```sh
pup finetune auth login baseten
```

Read `doctor`'s `checks` list (`name`, `ok`, `detail`, `required`); `ok` is false and the exit
code 1 when a required check fails. Relay each failed `detail`, which carries its fix.

- `pup`: pup is on `PATH` and at least `min_version`.
- `datadog_auth`: pup is signed in to Datadog. `train` and `deploy` make no Datadog calls, so
  a failure here does not block them.
- `credentials:baseten`: a Baseten key is found, and its source. Not required, but `train`,
  `deploy` and `undeploy` need it once run with `--confirm`.
- `baseten:loops-sdk` and `rendering:transformers`: listed as not required, but `train` and
  sampler `deploy` need both. A version warning means an untested Loops SDK version.

## 1. Input data

`train` takes two files of canonical rows, one JSON object per line: `--train` and
`--validation`. Both are required, and validation needs at least one row that renders. If the
user has one file, ask before splitting it; keep rows that share `source.trace_id` in one file
and never put the same conversation in both.

```json
{"id": "row-1", "messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Reset my password."}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "send_reset_link", "arguments": "{\"user\": \"me\"}"}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "sent"}, {"role": "assistant", "content": "I sent you a reset link."}], "tools": [{"type": "function", "function": {"name": "send_reset_link", "description": "Email a reset link.", "parameters": {"type": "object", "properties": {"user": {"type": "string"}}}}}]}
```

- Fields: `id` (required), `messages` (OpenAI chat format, roles `system`, `user`,
  `assistant`, `tool`), and optional `tools`, `source` and `schema_version` (`1`). Unknown
  fields are refused.
- Assistant messages may carry `tool_calls` (`arguments` is a JSON object encoded as a string),
  `reasoning_content`, and `weight`: `0` keeps the turn as context without training on it.
  Tool messages need `tool_call_id`.
- Structure: a system message only first; every tool result answers an earlier, unanswered
  call; tool call ids are unique; when `tools` is set, every called tool is declared; the last
  message is from the assistant.

Rows are rejected whole and never truncated or repaired, in two ways:

- **Schema.** The first row that breaks the schema stops the command (exit 2) with
  `<file>:<line>: <code>: <detail>`. Codes: `no_assistant_message`, `system_not_first`,
  `tool_result_without_call`, `duplicate_tool_call_id`, `tool_not_declared`,
  `last_message_not_assistant`, `invalid_arguments_json`, `invalid_row`. Fix or remove the row.
- **Rendering.** The model's chat template turns each trained assistant turn into one datum. A
  training row that cannot render is left out and listed in `rejected_train_rows` with `id`,
  `code` and `detail` (plus `tokens` and `limit` for `exceeds_context`). Any validation row that
  cannot render stops the command. Codes: `no_trainable_turn`, `template_failed`,
  `prefix_mismatch`, `no_target_tokens`, `exceeds_context`, `reasoning_unsupported`,
  `invalid_tokens`.

## 2. Plan (preview)

```sh
pup finetune train --train train.jsonl --validation validation.jsonl
pup finetune train --train train.jsonl --validation validation.jsonl --model glm-5p3-flash \
  --max-spend-usd <usd> --hourly-rate-usd <usd-per-hour>
```

The preview is free and needs no API key; it loads the model's tokenizer from Hugging Face.
Models: `qwen3p5-9b` (default) and `glm-5p3-flash`. Other flags: `--epochs` (5),
`--learning-rate` (1e-4), `--batch-size` (32 datums per optimizer step), `--lora-rank` (the
model's, 8), `--max-seq-len`, `--seed` (42), `--early-stopping-patience` (1), `--run-dir`,
`--run-id`. Change them only when the user asks.

Check in `preview`, and show the user:

- `dataset.rejected_train_rows`: how many and which codes. With `exceeds_context` and no
  `--max-seq-len`, those rows are longer than the model's context and cannot train.
- `dataset.train_rows`, `validation_rows`, `train_datums` and `validation_datums`.
- `config.max_seq_len` and `config.max_seq_len_source`: `data` means the longest rendered datum
  rounded up to a power of two (at least 1024); `flag` means `--max-seq-len`, and rows over it
  are rejected. The trainer is sized to it, and `deploy` serves it by default.
- `estimates.optimizer_steps_per_epoch` and `max_optimizer_steps`. A very small number means
  few updates; ask the user whether to lower `--batch-size`.
- `budget`: without the guard there is no cap, and the run lasts up to `max_epochs` epochs.
  With `--max-spend-usd` and `--hourly-rate-usd` (always both, from the user's own price),
  `maximum_active_seconds` is the wall-clock limit: 90% of the cap at that rate. It measures
  time, not the bill, and cannot stop a call in progress.
- pup-finetune has no price list, so the preview has no dollar estimate. Say that training
  provisions a dedicated Baseten GPU trainer that bills until the run is deactivated.

Ask the user to authorize the spend. To change a setting, preview again with the new flags.

## 3. Train

Once the user authorizes the spend, run the preview's `next_command` as printed. It pins the
generated run id and directory:

```sh
pup finetune train --train train.jsonl --validation validation.jsonl --provider baseten \
  --model qwen3p5-9b --run-id <run-id> --run-dir finetune/runs/<run-id> --confirm
```

Training runs in the foreground: several minutes to provision, then every epoch. Run it where
a tool timeout will not kill it (for example in the background with stderr saved to a file),
and start no second run in the same directory. Ctrl-C or SIGTERM still deactivates the trainer
and writes `result.json`; never use `kill -9`. Before anything is paid it checks that Baseten
offers the model at `max_seq_len` to this workspace.

stderr shows `epoch E/N step S/T: train loss X`, then
`epoch E/N: train loss X, validation loss Y (new best)`, and ends with
`training <status>; results in <run-dir>`. The run directory holds:

- `plan.json`: the plan that was trained.
- `epochs.json`: per epoch `train_loss`, `validation_loss`, `optimizer_steps`, `new_best`, and
  `sampler_weights_uri` on each new best.
- `run-state.json`: live status (`planned`, `provisioning`, `training`, then the final status),
  the Baseten session and run ids, and `cleanup_status`.
- `result.json`: the final result, written on failure too.

The best checkpoint is the epoch with the lowest validation loss: `best_epoch` and
`best_sampler_weights_uri` (`bt://loops:<run>/sampler_weights/<name>`). Early stopping ends
the run after `--early-stopping-patience` epochs without improvement (`stopped_early: true`).
`status` is one of:

- `completed`: all epochs ran, or early stopping ended the run. Exit 0.
- `budget_stopped`: the spend guard ended the run; the best checkpoint so far is kept. Exit 0.
- `failed`: an error stopped it. The command exits non-zero with no stdout result; read the
  `Error:` and `Hint:` lines and `result.json`.

On every exit the trainer is closed and its Baseten run deactivated. A stdout result means
cleanup succeeded (`cleanup.run_deactivated: true`). After a failure, read `cleanup` in
`result.json`: a `cleanup.error`, or a `could not deactivate` or `a run may remain` hint, means
a run may still bill. Relay the hint (the Baseten UI, or a `curl` command that reads
`$BASETEN_API_KEY` from the user's own environment) and get the user to act now. Report
validation loss in `epochs.json` that rises or never improves; do not change settings and
train again without the user.

## 4. Deploy

The train result's `next_command` previews a sampler deploy:

```sh
pup finetune deploy --run-dir finetune/runs/<run-id> --serving-mode sampler
pup finetune deploy --run-dir finetune/runs/<run-id> --serving-mode sampler --confirm
pup finetune deploy --run-dir finetune/runs/<run-id> --serving-mode dedicated --accelerator H100:1
```

- `sampler`: a Loops sampler loads the checkpoint; Baseten picks the GPUs, and `--accelerator`
  is refused.
- `dedicated` (the default): the checkpoint as its own Baseten model on exactly the GPUs of
  `--accelerator TYPE:COUNT` (required). It needs BDN volumes on the Baseten org.
- `--max-seq-len` defaults to the run's `max_seq_len` (at least 1024; at most the model's).
  `--timeout` is the wait for readiness in seconds (default 1800, at most 7200).
- Check the preview: `action` (`create`, or `resume` of the deployment this run directory
  records), `checkpoint_path`, `max_seq_len` and its source (`flag`, `run` or `model`), and
  `billing`. Ask the user to authorize it: it bills per hour until `undeploy`.

After `--confirm` it waits until the deployment is ready, then smoke-tests the OpenAI routes:
the model listing and its context, a text reply, a tool call and a reply to its result. The
result has `state: ready`, `openai_base_url`, `served_model`, `endpoint` (`model_id`,
`deployment_id`, `url`), `advertised_max_seq_len` and `cleanup_command`.

- Clients send `served_model` as the model name. In sampler mode it is the checkpoint path
  (`bt://loops:...`); in dedicated mode it is the checkpoint name. Requests authenticate with
  the user's Baseten API key.
- Relay the warning `replies include reasoning markup (</think>); clients must strip it`.
- A sampler that fails before it is ready is stopped. A failed dedicated deployment is kept:
  rerun the same deploy to resume it, or undeploy it.
- `endpoint.json` in the run directory is the deployment receipt; never edit it.

## 5. Undeploy

Preview, then run it with `--confirm` once the user authorizes:

```sh
pup finetune undeploy --run-dir finetune/runs/<run-id>
pup finetune undeploy --run-dir finetune/runs/<run-id> --confirm
```

It deactivates only the deployment that `endpoint.json` records, waits up to 120 s, and keeps
the checkpoint and the receipt. The result has `state: inactive`. Deploying the run again with
the same settings reactivates a dedicated deployment, or starts a new sampler.

## End state

Hand the user:

- the run directory, `run_id`, model, `status`, `best_epoch` with its validation loss, the
  epoch count, `stopped_early`, and the number of rejected training rows with their codes;
- confirmation that the trainer was deactivated (`cleanup.run_deactivated: true`);
- if deployed: `openai_base_url`, `served_model`, the endpoint ids, that it bills per hour,
  and the exact `cleanup_command`; or, once undeployed, `state: inactive`.

Nothing may bill without the user knowing: no active trainer, and every deployment either
acknowledged by the user or inactive.

## Debugging

Find the step and run directory, then read `run-state.json`, `result.json` and `endpoint.json`
before rerunning. stderr shows `Error: <message>` (`Error: baseten: ...` for Baseten failures)
and often `Hint: <fix>`.

| Message or symptom | Cause | Fix |
| --- | --- | --- |
| Exit 2 with `"status": "confirmation_required"` | A preview, not a failure | Show it; run `next_command` once the user authorizes |
| `... is a remote write and PUP_READ_ONLY is set` | `PUP_READ_ONLY=true` | Tell the user; only they can unset it |
| `no baseten API key found` | No `BASETEN_API_KEY` and nothing in the keychain | User runs `pup finetune auth login baseten`, or sets the variable |
| `there is no terminal to prompt for the key on` | `auth login` ran in an agent shell | The user runs it in their own terminal |
| `no OS keychain is available to store the key in` | No keychain (CI, headless Linux) | Set `BASETEN_API_KEY` in the environment |
| `Baseten rejected the API key`, or `... returned HTTP 401` or `403` | Wrong or revoked key | New key from the Baseten account settings, then `auth login` again |
| `could not read the baseten key from the OS keychain` | Keychain locked or access denied | Unlock it, or set `BASETEN_API_KEY` |
| `the Baseten Loops SDK is not installed`, `Hugging Face transformers is not installed` | Extra missing | `pip install 'pup-finetune[baseten]'` |
| `could not load the <repo> tokenizer at revision <rev>` | No access to huggingface.co and no cache | Check the network or the Hugging Face cache |
| `<file>:<line>: <code>: <detail>` | A row breaks the schema; the file is refused | Fix or remove that row (step 1) |
| `<file>:<line>: invalid JSON`, `expected a JSON object` | A line is not one JSON object | Fix the line |
| `N validation rows cannot be rendered for <model>` | Every validation row must render | Fix or remove them; the hint names the first |
| `no row of <file> can be rendered for <model>`, `<file> has no rows` | No usable rows | Read the codes in the hint; check the file and model |
| `invalid training settings: ...` | A flag value is out of range, or only one spend flag | Fix the flag; pass `--max-spend-usd` with `--hourly-rate-usd` |
| `unknown Baseten model '<alias>'` | Wrong `--model` | `qwen3p5-9b` or `glm-5p3-flash` |
| `<dir> is not empty` | The run directory was used before | Preview again for a fresh run id and directory; never clear the old one |
| `another pup-finetune process is using <dir>` | A train, deploy or undeploy holds the directory | Wait for it to finish |
| `Baseten Loops does not offer <model> to this workspace`, `this Baseten workspace is not enabled to train <model>` | Model not enabled; nothing was paid | Another `--model`, or ask Baseten to enable it |
| `Baseten trains <model> on at most N tokens, below this run's max_seq_len of M` | Workspace context limit; nothing was paid | Preview again with `--max-seq-len N`; longer rows are rejected |
| `Baseten has no capacity for the trainer (HTTP 429)` | No GPU capacity; nothing provisioned | Retry after the hinted wait, with a new run directory |
| `Baseten refused to create the trainer (HTTP <status>): ...` | Baseten refused the request (its reason follows); nothing was provisioned | Relay the reason; once the user resolves it, preview again for a new run directory |
| Hint `a run may remain in Loops session <id>` | Trainer creation was interrupted | The user deactivates it in the Baseten UI |
| `<VAR> is set; pup-finetune does not run Baseten Loops with it`, `LOOPS_BASE_URL must be unset or https://api.baseten.co` | A Loops SDK override in the environment | Unset it |
| `could not deactivate Baseten run <id>` | Cleanup failed; the trainer may still bill | The user deactivates it now (Baseten UI, or the hinted `curl`) |
| `training stopped: the process received SIGTERM` | The process was stopped; cleanup ran | Check `cleanup` in `result.json`; train again in a new directory |
| `<dir>/plan.json does not exist`, `<dir> is not a Baseten training run` | Wrong `--run-dir` | Pass the directory `train` wrote |
| `<dir> has no best sampler checkpoint to deploy` | No epoch finished | Nothing to deploy; train again |
| `--serving-mode dedicated needs --accelerator TYPE:COUNT, e.g. H100:1` | Dedicated without `--accelerator`, or not `TYPE:COUNT` (uppercase type) | Add `--accelerator H100:1` (or the user's GPUs), or use `--serving-mode sampler` |
| `--accelerator applies only to --serving-mode dedicated` | Sampler with GPUs | Drop `--accelerator` |
| `Baseten offers no single-node instance with exactly <accelerator>` | No such GPU shape | Pick one from the hint's list |
| `Baseten refused to create the deployment: ...` (BDN volumes) | The org has no BDN volumes; nothing was created | `--serving-mode sampler`, or ask Baseten to enable BDN volumes |
| `the checkpoint is still syncing on Baseten` | Upload in progress | Rerun deploy in a few minutes |
| `<dir>/endpoint.json records a different deployment of this run` | Another live deployment of this run | Undeploy it first |
| `... records no deployment IDs, so an earlier deploy has an unknown outcome` | An earlier create was interrupted | The user deactivates what the hint names in the Baseten UI; only after they confirm, delete `endpoint.json` (the one allowed run-file change) and preview deploy again |
| `the Baseten deployment was not ready after N s`, `the Baseten deployment is <STATUS>` | Slow or failed start | Dedicated: rerun deploy (larger `--timeout`) or undeploy. Sampler: already stopped; rerun deploy |
| `the deployment serves N tokens of context, below --max-seq-len M` | Smaller served context | Undeploy, then deploy with `--max-seq-len N` |
| `the tool smoke test ...`, `the text smoke test got no visible answer` | The endpoint fails a smoke test | Report it; undeploy a dedicated deployment the user does not want |
| `stopping the sampler failed too (...), so it may still bill` | Cleanup after a failed sampler deploy failed | Tell the user it may bill; preview `undeploy` and run its `next_command` once they authorize |
| `<dir> records no deployment` | Nothing was deployed from this directory | Pass the directory you deployed |
| `the Baseten deployment was not inactive after 120 s` | Slow deactivation | Rerun undeploy; if it persists, the Baseten UI |

When you report a failure, include the command, the run directory, the `Error:` and `Hint:`
lines, and any run, session, model or deployment ids.
