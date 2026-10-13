# pup-finetune

Fine-tune an open-weight model on how your agent already behaves. `pup-finetune` turns the LLM
spans your application sends to Datadog Agent Observability into a reviewed training dataset,
trains a LoRA on it (on Fireworks, or Baseten), compares the tuned model with the base model on
held-out conversations, and deploys it. It is a Pup extension, so you run it as
`pup finetune <command>`, and every Datadog read and write goes through Pup's OAuth login.

> **Status: early development.** Commands and file formats may change without notice.

## Quickstart

You need Python 3.12 or later, [uv](https://docs.astral.sh/uv/), and Pup 1.10.7 or later,
signed in to your Datadog organization with `pup auth login`. Training needs a Fireworks
account with access to Serverless Training, or a Baseten account.

```sh
git clone https://github.com/DataDog/pup-finetune.git && cd pup-finetune
uv sync --all-extras
pup extension install --local bin/pup-finetune --link \
  --description "Fine-tune a model on your Agent Observability traces"
pup finetune auth login fireworks    # hidden prompt; or set FIREWORKS_API_KEY
pup finetune doctor                  # what you can run now, and what to fix
```

**With a coding agent:** run `pup finetune skill install`, then ask:

> Use the datadog-finetune skill to fine-tune a model on the LLM spans of ml_app `<my-agent>`.

The agent shows you every paid or remote step before it runs, and waits for your yes.

**By hand:** `dataset preview` → `dataset pull` → review (`dataset sample` and `dataset
label`, or `dataset triage`) → `train` → `status` → `evaluate` → `deploy` → `undeploy`.

```sh
pup finetune dataset preview --ml-app my-agent --from 7d
pup finetune dataset pull datasets/my-agent --ml-app my-agent --from 7d
pup finetune dataset sample datasets/my-agent --count 20
pup finetune dataset label datasets/my-agent --file labels.jsonl
pup finetune train --dataset datasets/my-agent --model qwen3p8-27b --detach
pup finetune status --watch
pup finetune evaluate ft-20261007183443-a8b181
pup finetune deploy ft-20261007183443-a8b181
pup finetune undeploy ft-20261007183443-a8b181
```

Run the quickstart from a working directory of your own, or keep the paths above: a
`datasets/` or `runs/` that `pup-finetune` creates gets a `.gitignore`, since both hold trace
content.

Paid and remote commands only preview: they print what would run, where the data goes and
what bills, then exit `2`. The preview's `next_command` runs it, with `--confirm`. The
[safety table](docs/commands.md#safety) says what each command costs and where its data goes.

### Start here

| You have | Start at |
| --- | --- |
| An `ml_app` that sends LLM spans to Agent Observability | [Build a dataset](#build-a-dataset) |
| Conversations as JSONL rows ([format](docs/data-format.md)) | `pup finetune dataset import datasets/my-agent --file rows.jsonl`, then [Train](#train) |
| A dataset directory | [Train](#train) |
| A finished run | [Evaluate](#evaluate) or [Deploy](#deploy) |

### What each step needs

`doctor` checks all of this and says which steps are ready. A key comes from its environment
variable or from the OS keychain (`pup finetune auth login <provider>`). `uv sync --all-extras`
installs every extra. `evaluate` also uses the Datadog login to publish its experiments.

<!-- BEGIN GENERATED: what each step needs (scripts/gen_command_docs.py) -->
| Step | Datadog login | API key | Extra |
| --- | --- | --- | --- |
| `dataset preview`, `dataset pull` | yes | - | - |
| `dataset import`, `show`, `sample`, `label`; `status`, `runs` | - | - | - |
| `dataset triage` | - | `FIREWORKS_API_KEY` | - |
| `dataset push` | yes | - | - |
| `train`, `evaluate` on Fireworks | - | `FIREWORKS_API_KEY` | `fireworks`, `hf` |
| `deploy`, `undeploy` on Fireworks | - | `FIREWORKS_API_KEY` | `fireworks` |
| `train` on Baseten | - | `BASETEN_API_KEY` | `baseten`, `hf` |
| `evaluate` on Baseten | - | `BASETEN_API_KEY`, `FIREWORKS_API_KEY` | `baseten`, `hf` |
| `deploy`, `undeploy` on Baseten | - | `BASETEN_API_KEY` | - |
| `anthropic/<model>` judges | - | `ANTHROPIC_API_KEY` | `anthropic` |
| `openai/<model>` judges | - | `OPENAI_API_KEY` | - |
<!-- END GENERATED: what each step needs -->

### Reading results

Each command prints one JSON result on stdout, and progress, warnings and a summary on
stderr. Under a coding agent, Pup wraps it in an envelope whose `data` is the result. Pup's
flags go before `finetune`: `pup --output yaml finetune status`, `pup --jq .phase finetune
status`. Exit `0` is done, `2` a command to fix or a preview, `1` an outside failure. The files
under `files` hold the full record. `pup finetune help <command>` prints a command's help.

## Build a dataset

`dataset preview` converts the spans in memory and writes nothing. Tune the filter with it,
then pull:

```sh
pup finetune dataset preview --ml-app my-agent --from 7d --span-name chat --tag env:prod
pup finetune dataset pull datasets/my-agent --ml-app my-agent --from 7d --span-name chat --tag env:prod
pup finetune dataset show datasets/my-agent
```

The preview shows sample rows (the first user message and the final answer), how many rows it
would accept, and why it leaves the rest out. `--tag` repeats, and every tag must match.
`--query` takes a search query, and `--include-errors` keeps errored LLM calls. `--from` and
`--to` take ISO 8601, epoch milliseconds (a bare number is milliseconds), or a relative time:
`7d`, `24h`, `now-7d`, `now`.

The first pull freezes the filter and the time window, so `pup finetune dataset pull datasets/my-agent`
with no flags resumes it over the same window and limits. `--target-count` is the number of
rows to keep: rows a review drops do not count, so pulling again after a review reads on to
replace them. Every dataset command reports a `pipeline`
block (rows kept, dropped and unlabeled, triage, push), and its `next_command` follows from it.

### Review the rows

Read a sample and record which rows to keep. `labels.jsonl` holds one
`{"id": "<row id>", "keep": true, "reason": "..."}` per line; dropping a row needs a reason:

```sh
pup finetune dataset sample datasets/my-agent --count 20
pup finetune dataset label datasets/my-agent --file labels.jsonl
```

For a larger dataset, `dataset triage` has a council of judge models label the rows against
your rubric. It is paid, so it previews first:

```sh
pup finetune dataset triage --print-template > rubric.md    # then describe your agent's task
pup finetune dataset triage datasets/my-agent --rubric rubric.md --agreement
pup finetune dataset triage datasets/my-agent --rubric rubric.md
```

`--agreement` measures how often the council agrees with the rows you labeled, and labels
nothing. With the two default judges, a row is kept only when both vote keep; add a third
judge with `--judges` for a true majority. A row is dropped as soon as the votes cast decide
it (one drop of two judges, two of three), so its remaining calls are not made, and a failed
call does not leave it undecided.

### Push the dataset

Pushing creates an Agent Observability dataset with one record per kept row. Training does not
need it; publishing an evaluation as experiments does. The receipt, `push.json`, names the
dataset, so a later push of new rows needs no flag.

```sh
pup finetune dataset push datasets/my-agent --name my-agent-sft
```

The dataset in Datadog can then be curated there: records edited, added or removed in the
product. To train on it as it stands, fetch it into a new directory, which records the
dataset's version and gets its push receipt, so evaluations publish onto it:

```sh
pup finetune dataset fetch datasets/my-agent-v2 --project pup-finetune --name my-agent-sft
```

## Train

```sh
pup finetune train --dataset datasets/my-agent --model qwen3p8-27b --detach
pup finetune status --watch
```

The preview shows the rendered tokens per split, the rows left out (such as `exceeds_context`)
and where the rows go. The provider follows from the model, and the run goes in
`runs/<run id>`. `--detach` trains in a background worker (its log is `worker.log`). Cap the
spend with `--max-train-tokens`, or `--max-spend-usd` with `--hourly-rate-usd`. The epoch with
the best validation loss is kept, also when a cap stops the run: promoted to a Fireworks model,
or recorded as the run's `deployable_checkpoint` on Baseten, which has no promotion. The
Baseten models are `qwen3p5-9b` and `glm-5p3-flash`:
`pup finetune train --dataset datasets/my-agent --model qwen3p5-9b --detach`.

`--remote` (Baseten only) runs the training loop in a Baseten CPU job, so training continues
while this machine sleeps; `--team` names the Baseten team (by default your only or default
team). It needs the `baseten` CLI and `uv` on `PATH`, pup-finetune installed from a source
checkout, and a team secret `pup_finetune_api_key` holding a Baseten API key of your workspace.
The job bills on top of the trainer. `status` reads the job and its logs and copies the run's
state into the run directory, so `status --watch`, `evaluate` and `deploy` work as for a local
run. `cancel` cannot stop a remote run yet: stop its job, and its Loops trainer, in the Baseten
UI.

`pup finetune runs` lists your runs. Run commands take a run id or a run directory, and
`status` alone shows the newest run. `status --watch` prints progress until training ends;
Ctrl-C stops only the watch. `pup finetune resume ft-20261007183443-a8b181` continues an
interrupted run from its last checkpoint, and `cancel` stops one, keeping its best epoch.

### Choose the base model

Before training, compare candidate base models on the test split. Each one replays it with
the judge `evaluate` uses (paid; it previews first):

```sh
pup finetune benchmark datasets/my-agent --models qwen3p5-9b,glm-5p3-flash
```

The result ranks the models by judge pass rate with a confidence interval and recommends one
to fine-tune. A base model at or above `--target-pass-rate` (90% by default) may need no
training, and models within noise of the best are named, so the cheapest to serve can win.
Results go in `datasets/my-agent/benchmark`; rerun the command to continue one that stopped.

### A cheap first run

One epoch on a token budget, evaluated on one turn per test conversation. It checks the
pipeline, not quality. Set `--max-train-tokens` above the preview's `tokens_per_epoch`: below
it no epoch finishes, so nothing is promoted or evaluated (the preview warns):

```sh
pup finetune train --dataset datasets/my-agent --model qwen3p8-27b \
  --max-epochs 1 --max-train-tokens 200000 --evaluate --eval-max-cases 1
```

## Evaluate

```sh
pup finetune evaluate ft-20261007183443-a8b181
```

Evaluation replays each held-out test conversation with the base and the tuned model, and an
LLM judge decides whether each answer is an acceptable substitute for the recorded one. The
result leads with its `headline`:

```json
"headline": {
  "judge_pass_rate": {"base": 0.0, "tuned": 0.25, "delta": 0.25, "ci95": [0.0, 0.5]},
  "paired": {"wins": 1, "regressions": 0, "ties": 3},
  "cases": 4,
  "trajectories": 2
}
```

An interval (`ci95`) that includes 0 is not a clear difference. The result also has each
tool-call metric per model and, when the dataset was pushed, the two Agent Observability
experiments it published. They are created when the evaluation starts, which prints a link to
compare them (`publication.comparison_url`), and fill in as cases finish. Replay measures
agreement with the recorded behavior, not task success. Rerun the command to continue an
evaluation; other settings need a new `--out-dir`.

A generation cut off at `--max-tokens` fails, so the result counts them per model
(`headline.truncated`) and warns. A reasoning base model can spend the whole budget thinking
while a tuned model that learned to answer directly does not: raise `--max-tokens` before
you read the delta.

## Deploy

```sh
pup finetune deploy ft-20261007183443-a8b181
pup finetune undeploy ft-20261007183443-a8b181
```

Both preview first; the preview's `next_command` adds `--confirm`. `deploy --confirm` creates
an on-demand deployment on Fireworks, waits until it serves and sends one chat completion as a
smoke test. It bills per GPU-second while a replica runs (it
scales to zero when idle) until you undeploy it. The result's `usage` gives the base URL, the
model to request and an example request. Undeploying keeps the model.

On Baseten, `deploy` serves the run's `deployable_checkpoint` as a dedicated model on exactly
the GPUs of `--shape TYPE:COUNT` (default `H100:1`); the workspace needs BDN volumes and an
`hf_access_token` secret. It scales between 0 and 1 replicas and bills per GPU-second while a
replica runs, until you undeploy it. Undeploying deletes that model and keeps the checkpoint.

## Credentials and judges

`pup-finetune` never asks for Datadog API or application keys: Datadog access is Pup's. A
provider key comes from its environment variable or the OS keychain (for Baseten, without
either, the key of your default Baseten remote in `~/.trussrc`, read only):

```sh
pup finetune auth login fireworks    # hidden prompt; also baseten, anthropic, openai
pup finetune auth status             # where each key comes from, never the key
pup finetune auth logout fireworks
```

A key is never a flag value, never printed and never written to a file. Without an OS
keychain, use the environment variable.

LLM judges grade `evaluate` and `dataset triage`. They run on Fireworks by default, with the
same key. Name a judge `anthropic/<model>` or `openai/<model>` to judge on Anthropic or
OpenAI with that provider's key, and a triage council can mix them:

```sh
pup finetune evaluate ft-20261007183443-a8b181 --judge-model anthropic/claude-opus-5-5
pup finetune dataset triage datasets/my-agent --rubric rubric.md --judges deepseek-v4p1-flash,openai/gpt-6-astra
```

Anthropic judges need the `anthropic` extra. Without an API key, they sign in the way the
Anthropic SDK does on its own (`ANTHROPIC_AUTH_TOKEN`, or `ant auth login`), which
`auth status` reports as `anthropic-sdk`.

## Data handling

Dataset and run directories hold trace content: conversations, tool calls and tool results.
`pup-finetune` creates them with `0700` permissions and their files with `0600`. When it
creates `runs/` or `datasets/` in the working directory, it adds a `.gitignore` there so they
stay out of version control.

Training, `evaluate` and `dataset triage` send conversation content to the providers you
choose, and every preview lists each recipient in `data_recipients`. `PUP_READ_ONLY=true` (or
`pup --read-only`) refuses `--confirm` on `train`, `resume`, `evaluate`, `dataset push`,
`deploy` and `undeploy`; previews still run. `dataset triage` writes nothing to Datadog, so it
stays allowed, and it still sends rows to the judges. `train --remote` uploads the train and
validation rows, their split assignments, the run files and pup-finetune's source to Baseten
storage for its job; the test split and dropped rows stay on this machine.
[docs/data-format.md](docs/data-format.md)
describes the rows, the exclusion codes and the files in a dataset directory.

## Develop

[AGENTS.md](AGENTS.md) covers the layout, the conventions and an offline loop that trains on
a fake provider, with no network or credentials. [docs/architecture.md](docs/architecture.md)
explains how a provider plugs in, and [docs/commands.md](docs/commands.md) documents every
command and flag. See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute and
[SECURITY.md](SECURITY.md) to report a vulnerability.

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
