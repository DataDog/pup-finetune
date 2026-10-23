# pup-finetune

[![CI](https://github.com/DataDog/pup-finetune/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DataDog/pup-finetune/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

pup-finetune fine-tunes an open-weight model on your agent's own conversations. It reads the
LLM spans your application sends to Datadog Agent Observability and turns them into a training
dataset you review. It then trains a LoRA adapter on Baseten or Fireworks, compares the tuned
model with the base model, and deploys it.

pup-finetune is a Pup extension, so you run it as `pup finetune <COMMAND>`. Every Datadog read
and write goes through Pup's login.

> **Early development.** Commands and file formats may change. Report problems in
> [GitHub issues](https://github.com/DataDog/pup-finetune/issues).

## How it works

```text
dataset pull → review → dataset push → train → evaluate → deploy
```

Start at the step that matches what you have:

| You have | Start at |
| --- | --- |
| An `ml_app` that sends LLM spans to Agent Observability | [Find the spans and pull them](#1-find-the-spans-and-pull-them) |
| A dataset directory | [Review the rows](#2-review-the-rows) |
| JSONL rows, or a dataset in Agent Observability | [Start from data you already have](#start-from-data-you-already-have) |
| A reviewed, pushed dataset directory | [Train a model](#train-a-model) |
| A finished run | [Evaluate it](#evaluate-the-tuned-model) or [deploy it](#deploy-the-model) |

## Quickstart

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and, to train, a Baseten
or Fireworks account. To evaluate, benchmark or triage rows, you need a
[judge credential](#credentials) for Anthropic, Fireworks or OpenAI. On a Datadog site other
than `datadoghq.com`, read [Installation](#installation) first.

```bash
# Install Pup 1.10.7 or later, then sign in to Datadog in your browser
brew install datadog-labs/pack/pup
pup auth login

# Install pup-finetune as the `finetune` extension
git clone https://github.com/DataDog/pup-finetune.git && cd pup-finetune
uv sync --all-extras
pup extension install --local bin/pup-finetune --link \
  --description "Fine-tune a model on your Agent Observability traces"

# Store your training provider's key, with <PROVIDER> as baseten or fireworks
pup finetune auth login <PROVIDER>

# See which steps this setup can run, and what to fix
pup finetune doctor
```

### With a coding agent

Run `pup finetune skill install` where you start your agent, or add `-g` for every project.
Then ask:

```text
Use the datadog-finetune skill to fine-tune a model on the LLM spans of ml_app <ML_APP>.
```

The agent runs each step with you and asks before every paid or remote step.

### By hand

Replace `<ML_APP>` with your application's `ml_app`, `<PROVIDER>` with `baseten` or
`fireworks`, and `<RUN_ID>` with the run id that `train` prints. A step marked `preview` prints
what it would do and exits `2`. To go ahead, run its `next_command`.

```bash
# 1. Look at the spans, then pull them into a dataset directory
pup llm-obs spans search --ml-app <ML_APP> --span-kind llm --from 7d --limit 50 --summary
pup finetune dataset pull datasets/my-agent --ml-app <ML_APP> --from 7d

# 2. Read a sample, write your labels file (see Review the rows), then record it
pup finetune dataset sample datasets/my-agent --count 20
pup finetune dataset label datasets/my-agent --file datasets/labels.jsonl

# 3. Push the kept rows to an Agent Observability dataset (preview)
pup finetune dataset push datasets/my-agent --name my-agent-sft

# 4. Train in the background (preview), then follow it. For a first run, see Start with a cheap run
pup finetune train --provider <PROVIDER> --dataset datasets/my-agent --model qwen3p8-27b --detach
pup finetune status --watch

# 5. Compare the tuned model with the base model (preview)
pup finetune evaluate <RUN_ID>

# 6. Deploy the tuned model (preview), and undeploy it when you are done (preview)
pup finetune deploy <RUN_ID>
pup finetune undeploy <RUN_ID>
```

## Installation

On a Datadog site other than `datadoghq.com`, run `export DD_SITE=<SITE>`, such as
`datadoghq.eu`, before `pup auth login`, and add that line to your shell profile. Pup's README
covers [installing Pup](https://github.com/DataDog/pup#installation) without Homebrew, and
[Authentication](https://github.com/DataDog/pup#authentication) lists the sites and other ways
to sign in.

With `--link`, the extension is a link to `bin/pup-finetune`, which runs your checkout through
`uv run`. uv installs Python 3.12 if you don't have it.

### Update and remove

```bash
# Update from your checkout, with no reinstall since the extension is a link
git pull
uv sync --all-extras

# Delete your stored keys before you remove the extension, since they stay in the OS keychain
pup finetune auth logout anthropic
pup finetune auth logout baseten
pup finetune auth logout fireworks
pup finetune auth logout openai
pup extension remove finetune
```

[`pup extension upgrade`](https://github.com/DataDog/pup/blob/main/docs/EXTENSIONS.md) updates
only extensions installed from GitHub releases, so it does not update this one.

## Credentials

pup-finetune never asks for Datadog API or application keys, and never takes a key as a flag,
prints it or writes it to a file. A provider key comes from its environment variable or the OS
keychain. A Baseten key can also come from your default remote in `~/.trussrc`, which
pup-finetune reads and never writes.

`pup finetune auth login` stores a key for `anthropic`, `baseten`, `fireworks` or `openai` in the
OS keychain, from a hidden prompt. Without a keychain, set the environment variable instead.
`pup finetune auth status` shows where each key comes from, never the key.

LLM judges grade `evaluate` and `benchmark`, and they label rows in `dataset triage`. Without
`--judge-model` or `--judges`, an Anthropic credential picks Anthropic judges, else an OpenAI
credential picks OpenAI judges, else Fireworks models judge. With both an Anthropic and an
OpenAI credential, the default triage council has one judge of each, so triage sends each row
to both providers. [Choose the judge](docs/evaluate.md#choose-the-judge) names the models.

### What each step needs

`doctor` checks each of these, and `uv sync --all-extras` installs every extra. `evaluate` also
uses the Datadog login to publish its experiments.

<!-- BEGIN GENERATED: what each step needs (scripts/gen_command_docs.py) -->
| Step | Datadog login | API key | Extra |
| --- | --- | --- | --- |
| `dataset pull`, `fetch` | yes | - | - |
| `dataset import`, `show`, `sample`, `label`; `status`, `runs` | - | - | - |
| `dataset push` | yes | - | - |
| `train`, `evaluate`, `benchmark` on Baseten | - | `BASETEN_API_KEY` | `baseten`, `hf` |
| `deploy`, `undeploy` on Baseten | - | `BASETEN_API_KEY` | - |
| `train`, `evaluate`, `benchmark` on Fireworks | - | `FIREWORKS_API_KEY` | `fireworks`, `hf` |
| `deploy`, `undeploy` on Fireworks | - | `FIREWORKS_API_KEY` | `fireworks` |
| Judges for `dataset triage`, `evaluate` and `benchmark` | - | one of `ANTHROPIC_API_KEY`, `FIREWORKS_API_KEY` or `OPENAI_API_KEY` | `anthropic` for Anthropic |
<!-- END GENERATED: what each step needs -->

## Training providers

| | Baseten | Fireworks |
| --- | --- | --- |
| Your account needs | A workspace enabled to train the model on Loops. To deploy, BDN volumes and an `hf_access_token` secret. | Access to Serverless Training, and a key with access to exactly one account |
| API key | `BASETEN_API_KEY` | `FIREWORKS_API_KEY` |
| Training bills | A dedicated Loops trainer, for the GPU time it runs | A serverless training session, for the tokens it trains and samples |
| Best epoch kept as | `deployable_checkpoint`, a Loops checkpoint | `promoted_model`, a Fireworks model |
| `deploy` creates | A dedicated model on exactly the GPUs of `--shape TYPE:COUNT`, by default `H100:1` | An on-demand deployment on `--shape`, by default the first shape Fireworks validates |
| `undeploy` deletes | The model. The checkpoint stays. | The deployment. The promoted model stays. |

## Paid and remote steps

Without `--confirm`, a command that bills or changes something remote prints a preview and exits
`2`. The preview shows what would run, where the data goes and who bills for it. Its
`next_command` adds `--confirm` and pins what the preview resolved, such as `train`'s run id or
`evaluate`'s judge.

| Command | What `--confirm` does |
| --- | --- |
| `dataset triage` | Runs the judges and labels the rows (paid) |
| `dataset push` | Writes the kept rows to an Agent Observability dataset |
| `benchmark` | Replays and judges each base model (paid) |
| `train`, `resume` | Trains on the provider (paid) |
| `evaluate` | Replays and judges both models (paid), and publishes the experiments |
| `deploy` | Creates a deployment that bills until you undeploy it |
| `undeploy` | Deletes the deployment |

`pup --yes`, `DD_AUTO_APPROVE` and agent mode never stand in for `--confirm`.

With `PUP_READ_ONLY` set, or `pup --read-only`, previews still run, and every command in the
table but `dataset triage` refuses `--confirm`. Triage writes nothing to Datadog, so read-only
mode allows it, though it still sends rows to its judges. The
[safety table](docs/commands.md#safety) lists what each command costs and where its data goes.

## Build a dataset

### Start from data you already have

```bash
# Import JSONL rows in the canonical format into a new dataset directory
pup finetune dataset import datasets/my-rows --file rows.jsonl

# Or fetch the current records of an Agent Observability dataset, such as one you curated
pup finetune dataset fetch datasets/my-agent-v2 --project pup-finetune --name my-agent-sft
```

Imported rows follow the [row format](docs/data-format.md#the-canonical-row) and go on to
[Review the rows](#2-review-the-rows). A fetch writes a push receipt for the dataset's version,
so a fetched dataset goes on to [Train a model](#train-a-model), and evaluations publish onto it.

### 1. Find the spans and pull them

```bash
# Test a filter with Pup's span search, which writes nothing
pup llm-obs spans search --ml-app <ML_APP> --span-kind llm --span-name chat --tags env:prod \
  --from 7d --limit 50 --summary

# Pull the same spans into a dataset directory, then read its counts
pup finetune dataset pull datasets/my-agent --ml-app <ML_APP> --span-name chat \
  --tag env:prod --from 7d
pup finetune dataset show datasets/my-agent
```

Put your own span name and tags in place of `chat` and `env:prod`. Every `--tag` must match.
The pull reports why it left rows out, in `excluded_by_code` and `exclusion_explanations`.

The first pull freezes its source settings, so `pup finetune dataset pull datasets/my-agent`
alone resumes over the same window. If a few long traces could fill the dataset, set
`--max-rows-per-trace` or `--sample random` on the first pull, or pull into a new directory with
them. Rows you drop don't count toward `--target-count`, so a later pull reads more spans to
replace them.
[`dataset pull`](docs/commands.md#pup-finetune-dataset-pull) lists every option and time format.

### 2. Review the rows

Read a sample of rows, then write your decisions in a labels file, one line per row, with the
`id` that `dataset sample` shows. Dropping a row needs a reason, and a row without a label stays
in the dataset. `--markdown` gives each sampled row a labels line, ready to copy.

```json
{"id": "<ROW_ID>", "keep": false, "reason": "answers without calling the tool"}
```

```bash
# Read 20 rows in brief, and write them in full to a private Markdown file
pup finetune dataset sample datasets/my-agent --count 20 --markdown datasets/review.md

# Record the decisions in your labels file
pup finetune dataset label datasets/my-agent --file datasets/labels.jsonl
```

For a larger dataset, `dataset triage` has a council of judge models label the rows against
your rubric. A strict majority of keep votes keeps a row, so the default council of two judges
keeps a row only when both vote keep.

```bash
# Print the rubric template, then fill in your agent's task, what to keep and what to drop
pup finetune dataset triage --print-template > datasets/rubric.md

# Measure how often the council agrees with your labels, without labeling anything (preview)
pup finetune dataset triage datasets/my-agent --rubric datasets/rubric.md --agreement

# Label the other rows (preview)
pup finetune dataset triage datasets/my-agent --rubric datasets/rubric.md
```

`--checks checks.py` first runs your own Python checks on each row, free.
[Row checks](docs/data-format.md#row-checks) has the contract.

### 3. Push the dataset

`train` and `benchmark` need every kept row pushed to Agent Observability, and `evaluate`
publishes its experiments on that dataset.

```bash
# Add one record per kept row to an Agent Observability dataset (preview)
pup finetune dataset push datasets/my-agent --name my-agent-sft
```

The dataset goes in the `pup-finetune` project unless `--project` names another. The receipt,
`push.json`, names the dataset, so a later push needs no flags.

## Choose a base model

This step is optional. `benchmark` replays the test conversations with each candidate base model
and grades the answers with the judge `evaluate` uses.

```bash
# Compare two base models on 20 test conversations (preview)
pup finetune benchmark datasets/my-agent \
  --models <PROVIDER>/qwen3p8-27b,<PROVIDER>/glm-5p3-flash --max-rows 20
```

The result ranks the models by judge pass rate, with a confidence interval, and recommends one
to fine-tune. A base model at or above `--target-pass-rate`, 90% by default, may need no
training.

## Train a model

```bash
# Train in a background worker (preview), then follow the newest run
pup finetune train --provider <PROVIDER> --dataset datasets/my-agent --model qwen3p8-27b --detach
pup finetune status --watch
```

Both providers offer `qwen3p8-27b`, and [Supported models](#supported-models) lists the others.
The run goes in `runs/<RUN_ID>`, and `pup finetune runs` lists every run. `status --watch`
follows the run until training ends, and Ctrl-C stops only the watch. Under a coding agent, the
watch stops after 90 seconds while training goes on.

Training keeps the epoch with the best validation loss, even when a cap or `cancel` stops the
run. `--max-train-tokens` caps the trained tokens. `--max-spend-usd` caps a spend estimate from
your `--hourly-rate-usd`. `resume <RUN_ID>` continues an interrupted run, and
`cancel <RUN_ID>` stops a run that is training. The
[training guide](docs/train.md) covers caps, the learning-rate schedule, stopping and resuming.

On Baseten only, `--remote` trains in a Baseten CPU job that also bills, and `cancel` cannot stop
it. [Train in a Baseten job](docs/train.md#train-in-a-baseten-job) says what it needs and how to
stop it.

### Start with a cheap run

A first run on a small budget checks the pipeline end to end, though it says little about
quality.

```bash
# One epoch of at most 200,000 trained tokens, then one turn per test conversation (preview)
pup finetune train --provider <PROVIDER> --dataset datasets/my-agent --model qwen3p8-27b \
  --max-epochs 1 --max-train-tokens 200000 --evaluate --eval-max-cases 1
```

Set `--max-train-tokens` to at least the preview's `tokens_per_epoch`. A lower cap lets no
epoch finish, so the run keeps nothing to evaluate.

## Evaluate the tuned model

```bash
# Replay the test conversations with the base and tuned models (preview)
pup finetune evaluate <RUN_ID>
```

An LLM judge decides whether each answer is an acceptable substitute for the recorded one. The
result leads with its `headline`:

```json
"headline": {
  "judge_pass_rate": {"base": 0.0, "tuned": 0.25, "delta": 0.25, "ci95": [0.0, 0.5]},
  "paired": {"wins": 1, "regressions": 0, "ties": 3},
  "cases": 4,
  "trajectories": 2
}
```

`delta` is the tuned pass rate minus the base one. When its interval `ci95` includes 0, the
evaluation has not shown that the models differ. The tools are never run, so the scores measure
agreement with the recorded behavior, not task success.

An answer cut off at `--max-tokens` fails, and `headline.truncated` counts those per model. A
reasoning base model can spend its whole budget thinking and make the delta look larger than it
is. Before you trust the delta, evaluate into a new `--out-dir` with a higher `--max-tokens`.

When the dataset was pushed, the evaluation publishes two Agent Observability experiments, one
per model. It creates them when it starts, prints a link to compare them
(`publication.comparison_url`) and fills them in as conversations finish. Rerun the command to
continue an evaluation that stopped.

The [evaluation guide](docs/evaluate.md) covers the rest:

- [Choose the judge](docs/evaluate.md#choose-the-judge), and calibrate it
- [See why cases fail](docs/evaluate.md#see-why-cases-fail)
- [Reuse base answers across runs](docs/evaluate.md#reuse-base-answers-across-runs)
- [Compare with another model](docs/evaluate.md#compare-with-the-model-your-app-runs-today)
- [Add your own evaluators](docs/evaluate.md#add-your-own-evaluators), LLM judges and checks

## Deploy the model

```bash
# Deploy the run's model (preview), and delete the deployment when you are done (preview)
pup finetune deploy <RUN_ID>
pup finetune undeploy <RUN_ID>
```

`deploy --confirm` creates the deployment, waits until it serves and sends one smoke-test chat
completion. The result's `usage` gives the base URL, the model name and an example request. The
deployment scales between 0 and 1 replicas and bills per GPU-second while a replica runs, until
you undeploy it. [Training providers](#training-providers) shows what each provider creates.

A timeout or a failed smoke test can leave a deployment that bills. Follow the undeploy hint
even when the deploy failed.

## Supported models

`--model` takes the alias, the provider's model id or the Hugging Face name. For a model both
providers offer, name the provider with `--provider`, or as `<PROVIDER>/<MODEL>` in `benchmark`.
`pup finetune models list --provider baseten` and `pup finetune models list --provider fireworks`
show every name and context length.

<!-- BEGIN GENERATED: supported models (scripts/gen_command_docs.py) -->
| Model | `--model` | Baseten | Fireworks |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-9B` | `qwen3p5-9b` | yes | - |
| `zai-org/GLM-5.3-Flash` | `glm-5p3-flash` | yes | yes |
| `Qwen/Qwen3.8-27B` | `qwen3p8-27b` | yes | yes |
| `moonshotai/Kimi-K3` | `kimi-k3` | yes | yes |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | `deepseek-v4-flash-0731` | - | yes |
| `meta-models/Muse-Glimmer-30B` | `muse-glimmer-30b` | - | yes |
<!-- END GENERATED: supported models -->

## Data handling

- Dataset and run directories hold conversations, tool calls and tool results from your traces.
  A dataset's `replay_cache` also holds base model answers and the judge's reasons.
  pup-finetune creates the directories `0700` and their files `0600`.
- pup-finetune adds a `.gitignore` to the `datasets/` and `runs/` it creates. Keep the file
  `dataset sample --markdown` writes under `datasets/` too.
- Training, `benchmark`, `evaluate` and `dataset triage` send conversations to the providers you
  choose, every judge and any `--compare-model` included. Previews list them in
  `data_recipients`.
- `train --remote` uploads the train and validation rows with their split assignments, the run
  files and pup-finetune's source to Baseten storage. The test conversations and dropped rows
  stay on your machine.
- Row checks and evaluator checks are your Python code. They run locally with your privileges
  and send nothing.
- Tokenizers and chat templates come from Hugging Face at a pinned commit. Kimi K3's tokenizer
  is Python code from its repository (`trust_remote_code`), so rendering for `kimi-k3` runs that
  code locally. The `train`, `evaluate` and `benchmark` previews warn about it.

## Output and exit codes

Each command prints one JSON result on stdout, and its progress, warnings and a summary on
stderr. A few commands print text, such as `help` and `--version`. The result's `next_command`
is the command to run next, exactly as written, or `null`. Pup 2.0 and later wrap the result in
Pup's envelope, under `data`. Pup's flags go before `finetune`, as in
`pup --output yaml finetune status`.
[What a command prints](docs/data-format.md#what-a-command-prints-and-what-its-files-keep) covers
the envelope and `--no-envelope`.

Exit code `0` means the command finished. `2` means a preview, or a mistake you can fix in the
command or the directory. `1` means something outside failed, or local state is broken. An error
prints nothing on stdout. It prints `Error [<code>]: <message>` on stderr, and a hint when there
is one. A command that returns a result still prints it on stdout when it exits `1`. Two
examples are `doctor` when a required check fails and `deploy` when its smoke test fails.

[Exit codes and errors](docs/commands.md#exit-codes-and-errors) lists every error code.
`pup finetune help <COMMAND>` prints a command's help.

## Documentation

| Topic | Read |
| --- | --- |
| Every command and flag, the exit codes and the error codes | [docs/commands.md](docs/commands.md) |
| Rows, exclusion codes, what each command prints and the files it writes | [docs/data-format.md](docs/data-format.md) |
| Spend caps, the learning-rate schedule, stopping and resuming | [docs/train.md](docs/train.md) |
| Judges, reused base answers, compare models and evaluators | [docs/evaluate.md](docs/evaluate.md) |
| How the pieces fit, and how a provider plugs in | [docs/architecture.md](docs/architecture.md) |
| Contributing a change, and an offline loop for working on the code | [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md) |
| Reporting a vulnerability | [SECURITY.md](SECURITY.md) |
| Installing Pup, Datadog sites and Pup's commands | [Pup's README](https://github.com/DataDog/pup) |

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
