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

**By hand:** `dataset pull` → review (`dataset sample` and `dataset label`, or `dataset
triage`) → `dataset push` → `train` → `status` → `evaluate` → `deploy` → `undeploy`.

```sh
pup llm-obs spans search --ml-app my-agent --span-kind llm --from 7d --limit 50 --summary
pup finetune dataset pull datasets/my-agent --ml-app my-agent --from 7d
pup finetune dataset sample datasets/my-agent --count 20
pup finetune dataset label datasets/my-agent --file labels.jsonl
pup finetune dataset push datasets/my-agent --name my-agent-sft
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
| Conversations as JSONL rows ([format](docs/data-format.md)) | `pup finetune dataset import datasets/my-agent --file rows.jsonl`, then [Push the dataset](#push-the-dataset) |
| A dataset directory | [Push the dataset](#push-the-dataset), then [Train](#train) |
| A dataset in Agent Observability | `pup finetune dataset fetch`, then [Train](#train) |
| A finished run | [Evaluate](#evaluate) or [Deploy](#deploy) |

### What each step needs

`doctor` checks all of this and says which steps are ready. A key comes from its environment
variable or from the OS keychain (`pup finetune auth login <provider>`). `uv sync --all-extras`
installs every extra. `evaluate` also uses the Datadog login to publish its experiments.

<!-- BEGIN GENERATED: what each step needs (scripts/gen_command_docs.py) -->
| Step | Datadog login | API key | Extra |
| --- | --- | --- | --- |
| `dataset pull` | yes | - | - |
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

Look at the spans with pup first, and test a filter there, then pull:

```sh
pup llm-obs spans search --ml-app my-agent --span-kind llm --span-name chat --tags env:prod --from 7d --limit 50 --summary
pup finetune dataset pull datasets/my-agent --ml-app my-agent --from 7d --span-name chat --tag env:prod
pup finetune dataset show datasets/my-agent
```

The pull reports how many rows it accepted, and why it left the rest out
(`excluded_by_code`, with `exclusion_explanations`). `--tag` repeats, and every tag must match.
`--query` takes a search query, and `--include-errors` keeps errored LLM calls. `--from` and
`--to` take ISO 8601, epoch milliseconds (a bare number is milliseconds), or a relative time:
`7d`, `24h`, `now-7d`, `now`. So that a few long traces do not fill the dataset,
`--max-rows-per-trace N` keeps at most N rows of any trace, and `--sample random` reads every
span up to `--max-candidates`, then deals `--target-count` rows round-robin over the traces
in an order `--seed` fixes, instead of keeping the first rows read.

The first pull freezes the filter, these options and the time window, so `pup finetune dataset pull datasets/my-agent`
with no flags resumes it over the same window and limits. `--target-count` is the number of
rows to keep: rows a review drops do not count, so pulling again after a review reads on to
replace them. Every dataset command reports a `pipeline`
block (rows kept, dropped and unlabeled, triage, push), and its `next_command` follows from it.

### Review the rows

Read a sample and record which rows to keep. `--markdown` also writes the sampled
conversations in full to a private file, each with its labels line to copy. `labels.jsonl`
holds one `{"id": "<row id>", "keep": true, "reason": "..."}` per line; dropping a row needs
a reason:

```sh
pup finetune dataset sample datasets/my-agent --count 20 --markdown datasets/review.md
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
nothing. `--rows FILE` (one row id per line) reviews only those rows, in either mode. With a
council of two judges, a row is kept only when both vote keep; add a third judge with
`--judges` for a true majority. A row is dropped as soon as the votes cast decide it (one drop
of two judges, two of three), so its remaining calls are not made, and a failed call does not
leave it undecided.

### Push the dataset

Pushing creates an Agent Observability dataset with one record per kept row. `train` and
`benchmark` read only a dataset whose kept rows are all pushed, and an evaluation publishes
its experiments on it. The receipt, `push.json`, names the dataset, so a later push of new
rows needs no flag.

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

Training uses Adam on a LoRA adapter: `--learning-rate` (1e-4), `--batch-size` rows per step,
up to `--max-epochs` epochs with early stopping after `--patience` epochs without a
validation-loss gain, and `--lora-rank`. The rate is constant by default; `--lr-schedule
linear` or `cosine` decays it toward zero over the steps of `--max-epochs` epochs (the
preview's `max_steps`), after `--warmup-fraction` of them ramping up (`warmup_steps`). Early
stopping or a cap ends a run before its decay does: set `--max-epochs` to the length you
intend, and raise `--patience` to let the decay run. The preview's `settings` and `run.json`
record every value; the other Adam parameters are fixed (`settings.optimizer`).

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
`--max-rows 20` replays a seeded sample of 20 test conversations, for a quicker, cheaper
first comparison. Base answers already made on the dataset (see [Evaluate](#evaluate)) are
reused, and a model with every answer reused opens no session; a benchmark also shares its
answers, so evaluating the model it recommends reuses the base side. On a terminal, each
failed case says why on stderr (cut off at `--max-tokens`, an answer that could not be
parsed, or the judge's reason). Off a terminal, only the first failure of each cause gets a
line and the progress line counts the rest; set `PUP_FINETUNE_PROGRESS=steps` to see every
one.

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

Base answers are shared within a dataset: `evaluate`, `train --evaluate` and `benchmark` keep
the base model's answers and their judgments in `DATASET/replay_cache`, and a later
evaluation of the same dataset reuses those made with the same base model, renderer,
`--max-tokens` and context (judgments only with the same judge). Only the tuned model's
answers are then paid for, and no base sampler starts. The preview shows what is reused, from
which runs and since when (`reuse`); the result counts it per model (`reused`). Every run on
the dataset then shares one base sample, so their deltas compare directly but share its
noise. `--no-reuse` makes the answers again; deleting the directory only costs new answers.

A generation cut off at `--max-tokens` fails, so the result counts them per model
(`headline.truncated`) and warns. A reasoning base model can spend the whole budget thinking
while a tuned model that learned to answer directly does not: raise `--max-tokens` before
you read the delta.

### Compare with another model

```sh
pup finetune evaluate ft-20261007183443-a8b181 --compare-model anthropic/claude-opus-5-5
```

`--compare-model` replays the same cases with a third model, such as the one your app runs
today, and the same judge grades it. It is named as `--judge-model` is (an alias,
`anthropic/<model>`, `openai/<model>` or a Fireworks model id), and the judge must be another
model. The result names it as `compare_model` and adds `headline.tuned_minus_compare`, with
the same paired verdicts and interval, and publishing adds a third experiment to the same
comparison. Added to a complete evaluation, it reuses the base and tuned results and opens no
training session, so only the compare model's answers and their judgments are paid for. The
compare model may write up to 16,000 tokens, thinking included, at its provider's lowest
reasoning effort, and its answers are its own (no server-side fallback to another model).
Right before it is attached, one short call checks that it answers, so a mistyped model or a
rejected key fails before the directory keeps it or any training session opens. The
evaluation directory keeps its compare model: a rerun without the flag continues it, and
another model needs a new `--out-dir`. A case whose recorded conversation the compare model's
API refuses every time (on Anthropic and OpenAI, a tool call with no tool result right after
it, or a tool result with no call right before it; on Anthropic, also a conversation that
opens or ends on an assistant turn) is never sent: the preview counts it per reason
(`compare.skipped_by_reason`), the preview and the result warn, and it counts in none of the
compare model's rates, nor in `tuned_minus_compare`. A compare model that would be sent no
case at all is refused by the preview.

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

LLM judges grade `evaluate` and `dataset triage`. By default they follow the keys you have:
`evaluate` judges with `anthropic/claude-haiku-5-5` when an Anthropic key (or the Anthropic
SDK's login) is found, else `openai/gpt-5.4-mini` with an OpenAI key, else Fireworks'
`deepseek-v4p1-flash`. The triage council is `claude-haiku-5-5` and `gpt-5.4-mini` with both
keys, `claude-haiku-5-5` and `claude-sonnet-5-5` with Anthropic's, `gpt-5.4-mini` and
`gpt-6-astra` with OpenAI's, else two Fireworks models. The preview names the judges and where
the default came from, and a directory keeps the judges it was first judged with. Name a judge
`anthropic/<model>` or `openai/<model>` to choose one, and a triage council can mix them:

```sh
pup finetune evaluate ft-20261007183443-a8b181 --judge-model anthropic/claude-opus-5-5
pup finetune dataset triage datasets/my-agent --rubric rubric.md --judges deepseek-v4p1-flash,openai/gpt-6-astra
```

Anthropic judges need the `anthropic` extra. Without an API key, they sign in the way the
Anthropic SDK does on its own (`ANTHROPIC_AUTH_TOKEN`, or `ant auth login`), which
`auth status` reports as `anthropic-sdk`. A preview whose judge or compare model nothing
would sign in fails, naming the key and how to log in, rather than offer a `--confirm` that
would fail; with `PUP_READ_ONLY` set, when it offers no `--confirm` anyway, it warns instead.
A rerun with nothing left to judge needs no judge key.

## Data handling

Dataset and run directories hold trace content: conversations, tool calls and tool results.
A dataset's `replay_cache` also holds base model answers to its test conversations and the
judge's reasons, like an evaluation directory.
`pup-finetune` creates them with `0700` permissions and their files with `0600`. When it
creates `runs/` or `datasets/` in the working directory, it adds a `.gitignore` there so they
stay out of version control. `dataset sample --markdown FILE` writes the sampled conversations
to `FILE` (`0600`); keep it under `datasets/` so git ignores it.

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
