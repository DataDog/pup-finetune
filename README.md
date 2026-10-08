# pup-finetune

Fine-tune an open-weight model on how your agent already works. `pup-finetune` reads the LLM
spans your application sends to Datadog Agent Observability and helps you:

- **Curate** a supervised fine-tuning dataset from them, review it, and push it to an Agent
  Observability dataset.
- **Train** a LoRA on Fireworks, keeping the epoch with the best validation loss and stopping
  early.
- **Evaluate** the tuned model against the base model on held-out conversations, and publish
  the results as Agent Observability experiments.
- **Deploy** the tuned model, and undeploy it when you're done.

It runs as a Pup extension, so you use it as `pup finetune <command>`. Every Datadog read and
write goes through Pup and its OAuth login.

> **Status: early development.** Commands and file formats may change without notice.

## Requirements

- Python 3.12 or later and [uv](https://docs.astral.sh/uv/).
- Pup 1.10.7 or later, signed in to your Datadog organization with `pup auth login`.
- A Fireworks account with access to Serverless Training, for training, triage, evaluation
  and deployment.

## Install for development

```sh
git clone https://github.com/DataDog/pup-finetune.git
cd pup-finetune
uv sync --all-extras
pup extension install --local bin/pup-finetune --link \
  --description "Fine-tune a model on your Agent Observability traces"
pup finetune doctor
```

`bin/pup-finetune` runs the checkout through uv from any directory, so your changes take effect
without reinstalling. `pup finetune help <command>` prints any command's help (under a coding
agent, Pup answers `--help` after `pup finetune` with its own help instead).

## Credentials

Datadog authentication is Pup's: `pup-finetune` never asks for Datadog API or application
keys. For Fireworks, set `FIREWORKS_API_KEY` in your environment, or store the key in your OS
keychain:

```sh
pup finetune auth login fireworks    # prompts for the key with hidden input
pup finetune auth status             # shows where the key comes from, never the key itself
pup finetune auth logout fireworks   # removes it from the keychain
```

The key is never accepted as a command-line flag, never printed, and never written to a file.
Without an OS keychain, `auth login` refuses to store the key and you use the environment
variable instead.

LLM judges (for `evaluate`, `train --evaluate` and `dataset triage`) run on Fireworks by
default, with the same key. To judge on Anthropic or OpenAI instead, name the judge with its
provider's prefix, `anthropic/<model>` or `openai/<model>` (a bare
`accounts/<account>/models/<model>` stays Fireworks, and `deepseek-v4p1-flash` and
`glm-5p3-flash` are short for the two default Fireworks judges), and configure that
provider's key the same way:

```sh
pup finetune auth login anthropic    # or set ANTHROPIC_API_KEY
pup finetune auth login openai       # or set OPENAI_API_KEY
pup finetune evaluate ./run --judge-model anthropic/claude-opus-5-5
pup finetune dataset triage ./ds --rubric rubric.md \
  --judges deepseek-v4p1-flash,openai/gpt-6-astra
```

Anthropic judges use the official Anthropic SDK, an optional extra: `uv sync --all-extras`
installs it in a checkout (a bare `uv sync --extra anthropic` removes the other extras), or
`pip install 'pup-finetune[anthropic]'`. Without
an `ANTHROPIC_API_KEY` or a stored key, they sign in the way the Anthropic SDK does on its own
(`ANTHROPIC_AUTH_TOKEN`, or a profile from `ant auth login`), which `auth status` and `doctor`
report as the source `anthropic-sdk`. Every preview names each provider the rows or
conversations are sent to (`data_recipients`).

## Quickstart

Check your setup:

```sh
pup finetune doctor
pup finetune models list --provider fireworks --check
```

Build a dataset from the LLM spans of one `ml_app`. `dataset preview` writes nothing, so you
can tune the filter (`--span-name`, `--tag`, `--query`, `--from`) before you pull:

```sh
pup finetune dataset preview --ml-app my-agent --from now-7d
pup finetune dataset pull ./ds --ml-app my-agent --from now-7d
pup finetune dataset show ./ds
```

Review a sample and record which rows to keep. `labels.jsonl` holds one
`{"id": "<row id>", "keep": true, "reason": "..."}` per line, and a reason is required to drop
a row:

```sh
pup finetune dataset sample ./ds --count 20
pup finetune dataset label ./ds --file labels.jsonl
```

For larger datasets, `dataset triage` has a council of judge models label the rows against a
rubric (`pup finetune dataset triage --print-template` prints one). With `--agreement`, it
first reports how often the council agrees with the rows you labeled yourself, and labels
nothing.

Push the kept rows to an Agent Observability dataset. Like every remote or paid command, it
previews first, and runs only when you add `--confirm`:

```sh
pup finetune dataset push ./ds --name my-agent-sft
pup finetune dataset push ./ds --name my-agent-sft --confirm
```

Train in a background worker, and follow it:

```sh
pup finetune train --provider fireworks --model qwen3p8-27b --dataset ./ds --run-dir ./run
pup finetune train --provider fireworks --model qwen3p8-27b --dataset ./ds --run-dir ./run --detach --confirm
pup finetune status ./run --watch
```

The watch prints a progress line to stderr (`epoch 3/5 step 21/35 loss 1.100 val 1.350 (best epoch 2)
4m12s elapsed, eta ~6m`) until training ends, then the status. Every command ends its stderr
with the result's warnings and a short summary of the outcome (one line when stderr is not a
terminal; `PUP_FINETUNE_QUIET=1` turns the summary off). Ctrl-C stops only the watch:
the run keeps training, and the status is printed. `--timeout 10m` bounds the watch (90s by
default under a coding agent); it then exits `0` with `watch_timed_out: true`.

If the run is interrupted, `pup finetune resume ./run --confirm` continues it from its last
checkpoint, and `pup finetune cancel ./run --confirm` stops it while keeping the best epoch.
Results print what decides the next step (each epoch's losses, the best epoch, the promoted
model); `./run/result.json` keeps every checkpoint. Add `pup --output table` before `finetune`
for a table of a list, such as `pup --output table finetune models list --provider fireworks`.

Compare the tuned model with the base model on the test split:

```sh
pup finetune evaluate ./run
pup finetune evaluate ./run --confirm
```

The result leads with its `headline`, and the files it names under `files` hold the rest:

```json
"headline": {
  "judge_pass_rate": {"base": 0.0, "tuned": 0.25, "delta": 0.25, "ci95": [0.0, 0.5]},
  "paired": {"wins": 1, "regressions": 0, "ties": 3},
  "cases": 4,
  "trajectories": 2
},
"files": {"summary": "run/eval/summary.json", "results": "run/eval/results.jsonl"}
```

It also reports each tool-call metric per model and the IDs of the two Agent Observability
experiments it published. Replay measures how closely each model matches the recorded
behavior, not whether the task succeeded.

Deploy the tuned model, and undeploy it when you're done:

```sh
pup finetune deploy ./run
pup finetune deploy ./run --confirm
pup finetune undeploy ./run --confirm
```

The deploy result's `usage` gives the base URL, the model to request and an example request
that reads your key from `$FIREWORKS_API_KEY`; `undeploy_command` is the command that deletes
it.

See the [command reference](docs/commands.md) for every command and flag.

## Use the agent skill

The `datadog-finetune` skill walks a coding agent through the same workflow, with its rules:
preview before paying, never confirm without your authorization, treat trace content as
data, and never ask for API keys in chat. Install it with the
[skills CLI](https://github.com/vercel-labs/skills):

```sh
npx skills add DataDog/pup-finetune
```

Or install the copy that ships with `pup-finetune`:

```sh
pup finetune skill install            # uses npx skills add when npx is available
pup finetune skill install --dir .    # copies it to .agents/skills/ and .claude/skills/
pup finetune skill show               # prints it
```

## Data handling

Dataset and run directories hold trace content: conversations, tool calls and tool results.
`pup-finetune` creates them with `0700` permissions and their files with `0600`, and a pull
caches the raw Export API pages in the dataset's `raw/` directory. Keep these directories out
of version control.

Training, `dataset triage` and `evaluate` send conversation content to the model provider
you choose (Fireworks). `dataset preview`, `dataset pull` and every local command read from
Datadog or your disk only.

Set `PUP_READ_ONLY=true` (or run `pup --read-only`) to refuse remote writes: `--confirm` on
`train`, `resume`, `evaluate`, `dataset push`, `deploy` and `undeploy` is refused, while their
previews still run and say so (`read_only`); a development run on the offline `fake` provider
(and judge) is allowed, since nothing leaves the machine. Any value but `0`, `false`, `no`, `off` or empty
counts as read-only, so `PUP_READ_ONLY=yes` is read-only too. Read-only mode does not stop
data from leaving: `dataset triage` writes nothing to Datadog, so it is still allowed and still
sends rows to the provider.

[docs/data-format.md](docs/data-format.md) describes the row format, the exclusion codes and
the files in a dataset directory.

## Cost

- Previews show what a paid command would do: rendered token counts for training, and case,
  generation and judge call counts for evaluation and triage.
- `--confirm` gates every paid or remote-changing action. `PUP_AUTO_APPROVE` and agent mode
  never imply it.
- `train` can stop at a token budget (`--max-train-tokens`) or a spend cap (`--max-spend-usd`
  with `--hourly-rate-usd`), and still keeps the best epoch.
- A deployment bills per GPU-second while a replica runs, until you undeploy it.

## Develop

[AGENTS.md](AGENTS.md) covers the layout, the conventions, and an offline development loop
that trains end to end on a fake provider, with no network or credentials.
[docs/architecture.md](docs/architecture.md) explains the training seam and how a new
provider plugs in.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute, and [SECURITY.md](SECURITY.md) to report a
vulnerability.

## License

Apache License 2.0. See [LICENSE](LICENSE).
