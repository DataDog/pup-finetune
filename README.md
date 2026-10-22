# pup-finetune

`pup-finetune` fine-tunes an open-weight model on how your agent already behaves. It reads the
LLM spans your application sends to Datadog Agent Observability and turns them into a reviewed
training dataset. It trains a LoRA adapter on Fireworks or Baseten, compares the tuned model
with the base model on held-out conversations, and can deploy the tuned model.

It is a Pup extension, so you run it as `pup finetune <command>`. Every Datadog read and write
goes through Pup and its OAuth login.

> **Early development.** Commands and file formats may change without notice. Report problems
> in [GitHub issues](https://github.com/DataDog/pup-finetune/issues).

## Quickstart

### Install and check your setup

You need Python 3.12 or later, [uv](https://docs.astral.sh/uv/), and Pup 1.10.7 or later,
signed in to your Datadog organization with `pup auth login`. To train, you also need a
Fireworks account with access to Serverless Training, or a Baseten account.

```sh
git clone https://github.com/DataDog/pup-finetune.git && cd pup-finetune
uv sync --all-extras
pup extension install --local bin/pup-finetune --link \
  --description "Fine-tune a model on your Agent Observability traces"
pup finetune auth login fireworks    # hidden prompt; or set FIREWORKS_API_KEY
pup finetune doctor                  # what you can run now, and what to fix
```

`doctor` reports each step as ready or blocked, and gives the fix for each failed check.

### With a coding agent

Install the bundled `datadog-finetune` skill:

```sh
pup finetune skill install
```

Then ask your agent:

> Use the datadog-finetune skill to fine-tune a model on the LLM spans of ml_app `<my-agent>`.

The skill gives the agent the commands for every step below, in order. The agent shows you
each paid or remote step before it runs and waits for your yes.

### By hand

`dataset push`, `train`, `evaluate`, `deploy` and `undeploy` each print a preview and exit
`2`. To go ahead, run the command in the preview's `next_command`. Run the steps in this order:

```sh
pup llm-obs spans search --ml-app my-agent --span-kind llm --from 7d --limit 50 --summary
pup finetune dataset pull datasets/my-agent --ml-app my-agent --from 7d
pup finetune dataset sample datasets/my-agent --count 20
pup finetune dataset label datasets/my-agent --file labels.jsonl
pup finetune dataset push datasets/my-agent --name my-agent-sft
pup finetune train --provider fireworks --dataset datasets/my-agent --model qwen3p8-27b --detach
pup finetune status --watch
pup finetune evaluate ft-20261007183443-a8b181
pup finetune deploy ft-20261007183443-a8b181
pup finetune undeploy ft-20261007183443-a8b181
```

The paths are relative to your working directory. When `pup-finetune` creates a `datasets/`
or `runs/` directory, it adds a `.gitignore` to it, because both hold trace content.

### Start here

| You have | Start at |
| --- | --- |
| An `ml_app` that sends LLM spans to Agent Observability | [Build a dataset](#build-a-dataset) |
| Conversations as JSONL rows ([format](docs/data-format.md)) | `pup finetune dataset import datasets/my-agent --file rows.jsonl`, then [Push the dataset](#push-the-dataset) |
| A dataset directory | [Push the dataset](#push-the-dataset), then [Train](#train-a-model) |
| A dataset in Agent Observability | `pup finetune dataset fetch`, then [Train](#train-a-model) |
| A finished run | [Evaluate](#evaluate-the-tuned-model) or [Deploy](#deploy-and-undeploy) |

## Paid and remote steps preview first

Without `--confirm`, a command that bills or changes something remote prints a preview and
exits `2`. The preview shows what would run, where the data goes and who bills for it. Its
`next_command` is the same command with `--confirm`. It also repeats the values the preview
resolved, such as `train`'s `--run-id` and inferred `--provider`, or `evaluate`'s judge, so
the run matches the preview.

| Command | What `--confirm` does |
| --- | --- |
| `dataset triage` | Runs the judge models and labels the rows (paid) |
| `dataset push` | Writes the kept rows to an Agent Observability dataset |
| `benchmark` | Replays the test split with each base model and judges the answers (paid) |
| `train`, `resume` | Trains on the provider (paid) |
| `evaluate` | Replays the test split with the base and tuned models, judges the answers (paid) and publishes the experiments |
| `deploy` | Creates a deployment that bills while a replica runs, until you undeploy it |
| `undeploy` | Deletes the deployment |

`cancel` also asks for `--confirm`, though it only signals a local training process.

`--confirm` is never implied by `PUP_AUTO_APPROVE` or by agent mode. With `PUP_READ_ONLY` set,
previews still run, and `--confirm` is refused on paid runs and remote writes. The one
exception is `dataset triage`, which writes nothing to Datadog. Its `--confirm` stays allowed,
so it still sends rows to its judges, and its preview still needs their keys. The
[safety table](docs/commands.md#safety) lists what every command costs and where its data goes.

## Supported models

`--model` takes the alias, the provider's model id or the Hugging Face name. When both providers
offer a model, name the provider with `--provider`, or as `PROVIDER/MODEL` in `benchmark`.
`pup finetune models list` shows every name and each model's context length.

<!-- BEGIN GENERATED: supported models (scripts/gen_command_docs.py) -->
| Model | `--model` | Fireworks | Baseten |
| --- | --- | --- | --- |
| `Qwen/Qwen3.8-27B` | `qwen3p8-27b` | yes | yes |
| `zai-org/GLM-5.3-Flash` | `glm-5p3-flash` | yes | yes |
| `moonshotai/Kimi-K3` | `kimi-k3` | yes | yes |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | `deepseek-v4-flash-0731` | yes | - |
| `meta-models/Muse-Glimmer-30B` | `muse-glimmer-30b` | yes | - |
| `Qwen/Qwen3.5-9B` | `qwen3p5-9b` | - | yes |
<!-- END GENERATED: supported models -->

## What each step needs

`doctor` checks all of this, and `uv sync --all-extras` installs every extra. `evaluate` also
uses the Datadog login to publish its experiments.

The table names `FIREWORKS_API_KEY` for the judges of `dataset triage` and `evaluate`, because
Fireworks is their last fallback. An Anthropic or OpenAI key changes the default judge, as
[Choose the judge](docs/evaluate.md#choose-the-judge) explains.

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

## Build a dataset

Start here when your `ml_app` sends LLM spans to Agent Observability.

### Find the spans and pull them

Look at the spans with Pup first, and test your filter there. The search writes nothing. Then
pull the same spans into a dataset directory:

```sh
pup llm-obs spans search --ml-app my-agent --span-kind llm --span-name chat --tags env:prod --from 7d --limit 50 --summary
pup finetune dataset pull datasets/my-agent --ml-app my-agent --from 7d --span-name chat --tag env:prod
pup finetune dataset show datasets/my-agent
```

The pull reports how many rows it accepted and why it left the others out, in
`excluded_by_code` and `exclusion_explanations`. `--tag` repeats, and every tag must match. A
few long traces can fill a dataset, so to spread the rows over traces, add
`--max-rows-per-trace` or `--sample random`.

The first pull freezes its source settings: `--ml-app`, the span filter, the time window,
`--reasoning`, `--include-errors`, `--sample`, `--max-rows-per-trace` and the split settings.
Run `pup finetune dataset pull datasets/my-agent` with no flags to resume over the same window.

`--target-count` counts kept rows, so after a review drops some, pulling again reads more spans
to replace them. [`dataset pull`](docs/commands.md#pup-finetune-dataset-pull) lists every
option and the time formats `--from` and `--to` take.

### Review the rows

Read a sample, then record which rows to keep. `--markdown` also writes the sampled
conversations in full to a private file, each with a labels line to copy.

```sh
pup finetune dataset sample datasets/my-agent --count 20 --markdown datasets/review.md
pup finetune dataset label datasets/my-agent --file labels.jsonl
```

Each line of `labels.jsonl` is `{"id": "<row id>", "keep": true, "reason": "..."}`. Dropping a
row needs a reason.

For a larger dataset, `dataset triage` has a council of judge models label the rows against
your rubric. It is paid, so it previews first:

```sh
pup finetune dataset triage --print-template > rubric.md    # then describe your agent's task
pup finetune dataset triage datasets/my-agent --rubric rubric.md --agreement
pup finetune dataset triage datasets/my-agent --rubric rubric.md
```

Start with `--agreement`. It measures how often the council agrees with the rows you labeled,
and labels nothing. The default council has two judges, so a row is kept only when both vote
keep. Add a third judge with `--judges` for a true majority.

`--checks checks.py` runs your own Python functions on each row before any judge, free and on
this machine. A row that a check fails is labeled drop and never goes to a judge. `--checks`
alone needs no `--confirm`. [Row checks](docs/data-format.md#row-checks) has the contract.

### Push the dataset

Pushing creates an Agent Observability dataset with one record per kept row. `train` and
`benchmark` read only a dataset whose kept rows are all pushed, and `evaluate` publishes its
experiments on it. The receipt, `push.json`, names the dataset, so a later push needs no flags.

```sh
pup finetune dataset push datasets/my-agent --name my-agent-sft
```

You can then curate the dataset in Datadog by editing, adding or removing records. To train on
it as it is now, fetch it into a new directory. The fetch records the dataset's version and
writes its push receipt, so evaluations still publish onto it.

```sh
pup finetune dataset fetch datasets/my-agent-v2 --project pup-finetune --name my-agent-sft
```

## Choose a base model

This step is optional. Before you train, `benchmark` replays the test split with each
candidate base model and grades them with the judge `evaluate` uses.

```sh
pup finetune benchmark datasets/my-agent --models qwen3p5-9b,baseten/glm-5p3-flash
```

The result ranks the models by judge pass rate, with a confidence interval, and recommends one
to fine-tune. A base model at or above `--target-pass-rate` (90% by default) may need no
training at all.

`--max-rows 20` replays 20 test conversations for a cheaper first look. Results go in
`datasets/my-agent/benchmark`, and a rerun continues a benchmark that stopped. The evaluation
guide explains the [failure lines](docs/evaluate.md#see-why-cases-fail) that `benchmark` and
`evaluate` print on stderr.

## Train a model

```sh
pup finetune train --provider fireworks --dataset datasets/my-agent --model qwen3p8-27b --detach
pup finetune status --watch
```

The run goes in `runs/<run id>`. `--detach` trains in a background worker that logs to
`worker.log`, and `status --watch` follows it until training ends. Ctrl-C stops only the
watch.

Training keeps the epoch with the best validation loss, even when a cap stops the run. On
Fireworks, the run promotes it to a Fireworks model. Baseten has no promotion, so the run
records it as its `deployable_checkpoint`. Cap the spend with `--max-train-tokens`, or with
`--max-spend-usd` and `--hourly-rate-usd`.

The [`train` reference](docs/commands.md#pup-finetune-train) lists the training settings.
The learning rate stays constant unless `--lr-schedule linear` or `cosine` decays it. A
schedule decays over `--max-epochs` epochs, and early stopping or a cap can end the run first.
To let the decay finish, set `--max-epochs` to the length you want, raise `--patience`, and set
any cap above what those epochs need.

`pup finetune runs` lists your runs. Every run command takes a run id or a run directory, and
`status` alone shows the newest run. `pup finetune resume ft-20261007183443-a8b181` continues an
interrupted run from its last checkpoint. `cancel` stops a run and keeps its best epoch.

### Start with a cheap run

A first run on a small budget checks the pipeline end to end. It says little about quality.
This one trains one epoch on a token budget, then evaluates one turn per test conversation:

```sh
pup finetune train --provider fireworks --dataset datasets/my-agent --model qwen3p8-27b \
  --max-epochs 1 --max-train-tokens 200000 --evaluate --eval-max-cases 1
```

Set `--max-train-tokens` above the preview's `tokens_per_epoch`. Below it, no epoch finishes,
so nothing is promoted or evaluated, and the preview warns.

### Train on Baseten while your machine sleeps

```sh
pup finetune train --dataset datasets/my-agent --model qwen3p5-9b --remote
```

`--remote` runs the training loop in a Baseten CPU job, which bills on top of the trainer. Your
Baseten team needs a secret named `pup_finetune_api_key` that holds a Baseten API key of your
workspace.

`status` reads the job, so `status --watch`, `evaluate` and `deploy` work as for a local run.
`cancel` cannot stop a remote run, so stop its job and its Loops trainer in the Baseten UI.
The [`train` reference](docs/commands.md#pup-finetune-train) lists what else `--remote` needs.

## Evaluate the tuned model

```sh
pup finetune evaluate ft-20261007183443-a8b181
```

Evaluation replays each held-out test conversation with the base model and the tuned model.
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

When the interval `ci95` includes 0, the evaluation has not shown that the tuned model differs
from the base model. The tools are never run, so the scores measure agreement with the
recorded behavior, not task success. The result also has each tool-call metric per model.

When the dataset was pushed, the evaluation publishes two Agent Observability experiments. It
creates them when it starts and prints a link to compare them (`publication.comparison_url`).
They fill in as cases finish. Rerun the command to continue an evaluation. A changed
`--max-tokens`, judge, seed or `--max-cases` needs a new `--out-dir`.

An answer cut off at `--max-tokens` fails, and the result counts those per model in
`headline.truncated`. A reasoning base model can spend its whole budget thinking. Its cut-off
answers fail and make the delta look larger than it is, so raise `--max-tokens` before you
trust the delta.

### Reuse base answers across runs

A later evaluation or benchmark of the same dataset can reuse base answers from the dataset's
`replay_cache`. The preview's `reuse` shows what the run reuses. For when reuse applies and
what you then pay for, see
[Reuse base answers across runs](docs/evaluate.md#reuse-base-answers-across-runs) in the
evaluation guide.

### Compare with the model your app runs today

```sh
pup finetune evaluate ft-20261007183443-a8b181 --compare-model anthropic/claude-opus-5-5
```

`--compare-model` replays the same cases with a third model, graded by the same judge. Name it
the way you name a judge, and pick a model other than the judge. The result adds
`headline.tuned_minus_compare`, and publishing adds a third experiment.

Every answer comes from the model you name, because `pup-finetune` turns off Anthropic's
server-side fallback to another model for the compare model. For the cases a compare model's
API refuses, see
[Compare with the model your app runs today](docs/evaluate.md#compare-with-the-model-your-app-runs-today)
in the evaluation guide.

### Add your own evaluators

```sh
pup finetune evaluate ft-20261007183443-a8b181 --evaluators evals.toml
```

`--evaluators` scores every answer with your own LLM judges and Python checks too. The spec is
a TOML file:

```toml
headline = "all"                 # "judge" (the default), an evaluator's name, or "all"

[[evaluator]]
name = "polite"
type = "judge"
rubric = "Pass when the answer is courteous and does not blame the user."
examples = "polite-examples.jsonl"      # optional: answers you graded, to calibrate it

[[evaluator]]
name = "no_card_numbers"
type = "check"
file = "checks.py"
```

A check is a plain function of the case and the generated answer:

```python
def no_card_numbers(case, candidate):
    if any(char.isdigit() for char in candidate.get("content") or ""):
        return False, "the answer shows digits"
    return True
```

Added to a complete evaluation, evaluators score the stored answers, so only their judges'
calls are paid. `benchmark --evaluators` works the same way. An evaluation directory never
drops an evaluator, and it keeps each one as it was first added. To change one, leave its
entry as it is and add the new version under a new name, such as `polite_v2`. A changed check
goes in a new file.

Checks are Python you wrote, and they run in this process with your privileges. The preview
runs each check on every recorded answer, free, so a broken check fails before anything is
paid. [Evaluator specs and checks](docs/data-format.md#evaluator-specs-and-checks) has the
full spec. For calibration and the experiments, see
[Add your own evaluators](docs/evaluate.md#add-your-own-evaluators) in the evaluation guide.

## Deploy and undeploy

```sh
pup finetune deploy ft-20261007183443-a8b181
pup finetune undeploy ft-20261007183443-a8b181
```

Both preview first. On Fireworks, `deploy --confirm` creates an on-demand deployment, waits
until it serves and sends one chat completion as a smoke test. The deployment scales to zero
when idle and bills per GPU-second while a replica runs, until you undeploy it. The result's
`usage` gives the base URL, the model to request and an example request. Undeploying keeps
the model.

On Baseten, `deploy` serves the run's `deployable_checkpoint` as a dedicated model on exactly
the GPUs of `--shape TYPE:COUNT`, by default `H100:1`. The workspace needs BDN volumes and an
`hf_access_token` secret. The model scales between 0 and 1 replicas and bills per GPU-second
while a replica runs. Undeploying deletes the model and keeps the checkpoint.

## Keys live in your environment or the OS keychain

`pup-finetune` never asks for Datadog API or application keys, because every Datadog call
goes through Pup's login. A provider key comes from its environment variable or the OS
keychain. For Baseten, without either, it reads the key of your default Baseten remote in
`~/.trussrc`, and never writes it.

```sh
pup finetune auth login fireworks    # hidden prompt; also baseten, anthropic, openai
pup finetune auth status             # where each key comes from, never the key
pup finetune auth logout fireworks
```

A key is never a flag value, never printed and never written to a file. Without an OS
keychain, use the environment variable.

## Your credentials pick the default judges

LLM judges grade `evaluate` and `benchmark`, and they label rows in `dataset triage`. Without
`--judge-model` or `--judges`, an Anthropic credential picks Anthropic judges, else an OpenAI
credential picks OpenAI judges, else Fireworks models judge. With both an Anthropic and an
OpenAI credential, the triage council has one Anthropic judge and one OpenAI judge, so triage
sends each row to both providers. Name a judge to choose one, and mix providers in a triage
council:

```sh
pup finetune evaluate ft-20261007183443-a8b181 --judge-model anthropic/claude-opus-5-5
pup finetune dataset triage datasets/my-agent --rubric rubric.md --judges deepseek-v4p1-flash,openai/gpt-6-astra
```

For the defaults, the Anthropic SDK's own sign-in and a preview with no judge key, see
[Choose the judge](docs/evaluate.md#choose-the-judge) in the evaluation guide.

## Where your data goes

Dataset and run directories hold trace content, meaning conversations, tool calls and tool
results. A dataset's `replay_cache` also holds base model answers to its test conversations
and the judge's reasons, as an evaluation directory does. `pup-finetune` creates these
directories with `0700` permissions and their files with `0600`.

When `pup-finetune` creates `runs/` or `datasets/` in the working directory, it adds a
`.gitignore` there to keep them out of version control. `dataset sample --markdown FILE` writes
the sampled conversations to `FILE` with `0600` permissions. Keep that file under `datasets/`
so git ignores it.

Training, `benchmark`, `evaluate` and `dataset triage` send conversation content to the
providers you choose. `--compare-model` also sends it to the compare model's provider. Every
preview lists each recipient in `data_recipients`.

`PUP_READ_ONLY=true`, or `pup --read-only`, refuses `--confirm` on `dataset push`, `benchmark`,
`train`, `resume`, `evaluate`, `deploy` and `undeploy`. `dataset triage` writes nothing to
Datadog, so its `--confirm` stays allowed, and it still sends rows to the judges.

`train --remote` uploads the train and validation rows, their split assignments, the run files
and pup-finetune's source to Baseten storage. The test split and the dropped rows stay on this
machine. Checks run on this machine and send nothing. The evaluators' judges are recipients,
like the built-in judge.

Each model's own tokenizer and chat template render the rows. `pup-finetune` downloads both
from Hugging Face at a pinned commit. Kimi K3's tokenizer is Python code from the model's
repository (`trust_remote_code`, which `models list` shows). Rendering for `kimi-k3` downloads
that code at its pinned commit and runs it on this machine, and the `train`, `evaluate` and
`benchmark` previews warn about it.

## Every command prints one JSON result

Each command prints one JSON result on stdout, and progress, warnings and a summary on stderr.
Exit code `0` means the command finished. `2` means a preview, or a mistake you can fix in the
command or the directory. `1` means something outside failed, or local state is broken. Under
a coding agent, Pup wraps the result in an envelope whose `data` is the result. The files
listed under `files` hold the full record.

Pup's flags go before `finetune`, as in `pup --output yaml finetune status` or
`pup --jq .phase finetune status`. `pup finetune help <command>` prints a command's help.

## More docs

| To | Read |
| --- | --- |
| Choose judges, reuse base answers, compare models and add evaluators | [docs/evaluate.md](docs/evaluate.md) |
| Look up every command, flag, exit code and error code | [docs/commands.md](docs/commands.md) |
| Read the row format, the exclusion codes and the files a command writes | [docs/data-format.md](docs/data-format.md) |
| Learn how the pieces fit and how a provider plugs in | [docs/architecture.md](docs/architecture.md) |
| Work on the code, with an offline loop that needs no credentials | [AGENTS.md](AGENTS.md) |
| Contribute a change | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Report a vulnerability | [SECURITY.md](SECURITY.md) |

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
