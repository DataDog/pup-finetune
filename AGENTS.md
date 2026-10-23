# AGENTS.md

Guide for contributors and coding agents working in this repository.

## What this is

`pup-finetune` curates a supervised fine-tuning dataset from your Datadog Agent Observability
traces, fine-tunes an open-weight model on it, evaluates the tuned model against the base model,
and deploys it. It ships as a `pup` extension, so users run it as `pup finetune ...`; the
standalone executable is `pup-finetune`. All Datadog reads and writes go through `pup` and its
OAuth login. The project is in early development: commands and file formats may change.
User-facing docs: `README.md` (the flow and where to start, a quickstart, installation,
credentials, one table of the training providers' differences, how paid steps preview,
then one how-to section per step: dataset, base model, train, evaluate, deploy, then the
supported models, data handling and output; it links the reference docs and the guides),
`docs/commands.md` (every command and flag), `docs/data-format.md` (rows, exclusion codes,
dataset files, row checks, what a command prints), `docs/train.md` (a how-to guide to spend
caps, the learning-rate schedule, stopping and resuming, and `train --remote`),
`docs/evaluate.md` (a how-to guide to judges, reuse, the compare model and evaluators, with an
evaluator spec and check) and the agent skill in `skills/datadog-finetune/SKILL.md`.

## Commands

```sh
uv sync --all-extras                 # create .venv and install the project, extras + dev tools
uv run pytest                        # tests (no network, no paid calls)
uv run ruff check --fix              # lint
uv run ruff format                   # format (check only: uv run ruff format --check)
uv run mypy src                      # type-check (strict)
uv run python scripts/check_headers.py  # license header on every tracked .py and shell script
uv run python scripts/check_3rdparty.py # LICENSE-3rdparty.csv covers every dependency
uv run python scripts/gen_command_docs.py  # regenerate docs/commands.md, the exclusion
                                     # table in docs/data-format.md and README.md's tables of
                                     # supported models and of what each step needs; a test
                                     # fails when stale
uv run pre-commit install            # run the hooks above on every commit
uv run pup-finetune --help           # run the CLI from the checkout
```

The `hf` extra (`transformers`, `jinja2`, `huggingface-hub`) backs the shared renderer and each
provider SDK has its own extra (`baseten`, `fireworks`); CI installs every extra. CI
(`.github/workflows/ci.yml`) runs the lint, format, type, test, header and third-party license
checks on Ubuntu and macOS with Python 3.12 and 3.13, and smoke-tests the built wheel. When you
add a dependency, add its row to `LICENSE-3rdparty.csv`.

To develop against pup, install the checkout as a pup extension. `bin/pup-finetune` is a small
POSIX sh launcher that runs `uv run --project <checkout> pup-finetune` from any directory, and
pup names the extension `finetune` after the file:

```sh
pup extension install --local bin/pup-finetune --link \
  --description "Fine-tune a model on your Agent Observability traces"
pup finetune help                    # pup answers `pup finetune --help` itself under agents
pup finetune doctor                  # what this setup can run: pup, Datadog login, keys, SDKs
pup finetune doctor --provider fireworks --check   # its checks required, plus a live preflight
```

`doctor` reports each check's `status` (`pass`, `fail`, `optional`, `skipped`), `needed_for`
and `fix`, a `ready` map per step, `mode`, `versions` (the checkout's commit read from git at
run time) and a `next_command`, then a checklist on stderr. Only pup and the Datadog login
decide its `status` (and exit 1), unless `--provider` makes that provider's key and packages
required; `--check` adds its live, read-only preflight per model. Everything else is offline.
The top-level `ok` and each check's `ok` stay for one release.

The offline dev loop trains end to end on the `fake` provider, with no network or credentials.
`train` and `benchmark` on a real provider read only a dataset that is in Datadog: every kept
row pushed as it is now and no push in flight, or a `dataset fetch` directory
(`curation/pipeline.py`'s `require_in_datadog`, with its reason and step from
`Pipeline.push_gap`). Rows changed after the push are named, and their step fetches the
dataset as Datadog holds it, naming the rows dropped or retired since the push that the
fetch brings back. An offline provider (`ProviderInfo.offline`, the fake) is exempt, so this
loop and the default tests train on an imported directory that was never pushed:

```sh
export PUP_FINETUNE_DEV=1
uv run pup-finetune dataset import /tmp/ds --file tests/fixtures/rows/tool_use.jsonl
uv run pup-finetune train --model fake-model --dataset /tmp/ds --run-dir /tmp/run
                                     # preview only (exit 2): the provider is inferred from
                                     # --model; its next_command pins it and adds --confirm
uv run pup-finetune train --model fake-model --dataset /tmp/ds --run-dir /tmp/run --confirm
uv run pup-finetune status /tmp/run  # phase, steps, losses, best epoch, elapsed, approx ETA,
                                     # worker, promoted model
uv run pup-finetune benchmark /tmp/ds --models fake-model   # preview only (exit 2); with
                                     # --confirm, ranks base models on the test split into
                                     # /tmp/ds/benchmark and recommends one to fine-tune
uv run pup-finetune evaluate /tmp/run          # preview only (exit 2): cases and calls
uv run pup-finetune evaluate /tmp/run --confirm
                                     # replays the test split with base and tuned into
                                     # /tmp/run/eval; rerun to resume; summary.json at the end
uv run pup-finetune deploy /tmp/run  # preview only (exit 2): what is created, how to undeploy
uv run pup-finetune deploy /tmp/run --confirm    # receipt in /tmp/run/deployment.json;
                                     # usage (base URL, model, example) and undeploy_command
uv run pup-finetune undeploy /tmp/run --confirm  # deletes it; the model is kept
uv run pup-finetune evaluate /tmp/run --compare-model fake-compare --confirm
                                     # adds the offline compare target to the same
                                     # directory: compare pairs only, no training session
uv run pup-finetune evaluate /tmp/run --evaluators tests/fixtures/evaluators/evals.toml --confirm
                                     # adds a rubric judge and two checks: they score the
                                     # stored answers, nothing is generated again
```

An evaluation is also published to Datadog as two Agent Observability experiments
(`pup-finetune-base-<run id>` and `pup-finetune-tuned-<run id>`, plus
`pup-finetune-compare-<run id>` with a compare target) on the dataset `dataset push`
created, at the version it pushed. It creates them when it starts, prints a link comparing
them, and sends each test conversation once all its cases have a result (one with an error
waits for the end): a root span linked to its dataset record and a child span per replayed
action, with the metrics and judge verdicts. It needs the
dataset's push receipt (`push.json`): without one the preview and the result warn and the
evaluation is not published (the local results are complete either way); the result's
`next_command` is then `null`, and its `next_steps` push the dataset and rerun the evaluation,
which reuses every finished result (no new model calls when it is complete) and publishes it,
or deploy it unpublished. A complete evaluation that was published, or run with `--no-publish`
(`--eval-no-publish` for `train --evaluate`, which skips publishing without a warning), has the
`deploy` preview as its `next_command`. Publishing resumes:
`experiments.json` and `published.json` in the evaluation directory record the experiments
created and the spans sent, so a rerun creates and sends nothing twice, and a failed publish
exits 1 with a `next_command` that publishes the rest (a Ctrl-C while publishing leaves a
`cancelled` publication that the rerun continues too). Spans already published keep the
results they were sent with: retrying an evaluation's errors updates only the local results,
so evaluate into a new `--out-dir` for experiments with every result. Spans not indexed within
two minutes (one budget for both experiments) are a warning. Tests serve it with `FakePup` and `FakeExperimentsApi`; nothing reaches Datadog.

`evaluate` on a fake run uses the offline fake judge (`--judge-model fake-judge`, dev mode
only), and the fake's tuned checkpoint calls the tools its training data called, so the
result shows a positive `headline.judge_pass_rate.delta` (`tuned_minus_base` in `summary.json`).
Evaluating a second run of the same dataset reuses the first one's base answers and verdicts
from `/tmp/ds/replay_cache` (the preview's `reuse`; `--no-reuse` makes them again). `train ... --evaluate --confirm` evaluates in the
training session right after promotion instead (foreground only; `--eval-out-dir`, which must
be new or empty and is checked before training, `--eval-max-cases`,
`--eval-judge-model`, `--eval-concurrency`, `--eval-max-tokens`, `--eval-seed` and
`--eval-skip-calibration` configure it). The run's end is recorded before that evaluation
starts, so a crash during it keeps the trained, promoted run; a failed evaluation exits 1
with the trained run and the `evaluate` command to retry it. A cancelled or budget-capped
training is not evaluated in its session (the evaluation would spend the session's time past
the cap). A budget-capped run still ends `completed`, so `evaluate` evaluates it on request;
like a budget-capped run, a cancelled run is evaluated by `evaluate` on request (its best
epoch; one cancelled before any epoch has nothing to evaluate). Evaluating a real run is paid
and needs the judge's key (the default judge follows the keys there are; see below); the
preview is free and allowed with `PUP_READ_ONLY` set, evaluating is not (an offline one, on the
fake provider and judge and not published, is allowed: nothing leaves the machine; `train` and
`resume` on the fake provider likewise). A changed `--max-tokens`, judge, seed or cap needs a
new `--out-dir`. An evaluation whose pairs ended in errors (a sampler or judge outage) reports
`complete_with_errors` with an error count and a `next_command` that retries them, and exits
1 when a target has no verdict at all; a cancelled one reports `partial`. `evaluator_errors`
counts each evaluator's errors with the first one; a check that raised is not retried (its
frozen file raises again), and the hint names it with the fix.

To build a dataset from your own Agent Observability traces (Datadog reads through pup, so
both work with `PUP_READ_ONLY` set; test a filter with `pup llm-obs spans search` first):

```sh
pup llm-obs spans search --ml-app my-app --span-kind llm --from 7d --limit 50 --summary
uv run pup-finetune dataset pull /tmp/ds --ml-app my-app --from 7d
uv run pup-finetune dataset pull /tmp/ds   # rerun to resume: the frozen settings fill in
                                     # the flags (pages cached in /tmp/ds/raw)
uv run pup-finetune dataset show /tmp/ds
uv run pup-finetune dataset sample /tmp/ds -n 20              # rows in brief, to review
uv run pup-finetune dataset sample /tmp/ds --markdown /tmp/review.md  # and in full (0600)
uv run pup-finetune dataset label /tmp/ds --file labels.jsonl  # {id, keep, reason} per line
uv run pup-finetune dataset triage --print-template > rubric.md  # Task / Keep / Drop / Examples
uv run pup-finetune dataset triage /tmp/ds --rubric rubric.md --max-rows 20
                                     # preview only (exit 2): rows, judge calls, rubric hash;
                                     # add --confirm to have the judge council label them (paid)
uv run pup-finetune dataset triage /tmp/ds --checks tests/fixtures/evaluators/row_checks.py
                                     # labels the rows a check fails by: check, at once (free,
                                     # local, no --confirm); with --rubric they run first
uv run pup-finetune dataset push /tmp/ds --name my-dataset --project pup-finetune-dev
                                     # preview only (exit 2); add --confirm to push the kept rows
uv run pup-finetune dataset push /tmp/ds   # later pushes: push.json names the dataset
```

A relative `--from` or `--to` (`now`, `now-<n><unit>` or pup's `<n><unit>` with unit `s`,
`m`, `h`, `d` or `w`, or spelled out as pup takes them: `5min`, `2hours`, `'5 minutes'`) is
read into `now` or `now-<n><unit>` (`spans.parse_time_bound`; `7d` is `now-7d`) and resolved to
epoch ms when the command starts; a bare number is epoch milliseconds (pup reads seconds). The
first pull into a directory freezes that window together with the expressions, and a re-run
with the same expressions reads the frozen window, so a resumed pull reads the same spans. On
a re-run, every source flag left out (`--ml-app`, the filter, the time range, `--reasoning`,
`--include-errors`, `--sample`, `--max-rows-per-trace` and the split flags) takes the frozen
value, and one given with another value is a `settings_conflict` naming the frozen one;
`--ml-app` is required only for the first pull. The result's `window` holds the bounds read
and whether they are the frozen ones.

Every dataset command's result holds a `pipeline` block (`curation/pipeline.py`): `rows`
(`accepted`, `kept`, `dropped`, `unlabeled`), `triage` (`none`, `incomplete`, `complete`; see
`triage.triage_status`) and `push` (`status` `not_pushed`, `pushed`,
`pushed_with_new_rows`, `pushed_with_changed_rows` or `unreadable`, `pending` and `changed`),
which `cli/dataset.py`'s `dataset_next` turns into `next_command` and `next_steps`. A
validation or test split far from its fraction in a dataset of fewer than 50 traces is a
warning (`split.skew_warnings`, through `DatasetDir.summary`), which suggests more rows, never
another seed.

`dataset push` is a remote write: with `PUP_READ_ONLY` set its `--confirm` is refused (the
preview, built from local files, still runs). Push test data to the `pup-finetune-dev`
project, never to a shared one. A dataset's `next_steps` name `train` and `benchmark` only
once every kept row is pushed; until then they name the push.

`dataset triage` (`--rubric FILE` and/or `--rule TEXT`, repeated) has each judge model
(`--judges m1,m2`; by default chosen by the keys there are, see below) review
each whole row and vote keep or drop: a strict majority of keep votes keeps a row, a tie
drops it, a row missing a vote stays undecided. Votes go to `judgments.jsonl` as they arrive,
so rerunning the same command resumes and retries only failed calls; the rubric hash and
judges are frozen in `triage.json` once votes exist (`--reset-triage` discards them). Decided
rows are labeled `by: judge`; rows the agent or a person labeled are skipped. A row a judge
refuses as longer than its context window is dropped (`too_long`), never truncated, and its
other calls are skipped. `--rows FILE` (row ids, one per line, blank lines ignored; not frozen)
restricts the rows reviewed, before `--max-rows`, in both modes (`plan_triage`'s `only`); an id
that is not an accepted row is a usage error. `--agreement` reviews the agent's and people's
labeled rows instead, labels nothing, and reports the council's agreement with those labels
(rate, Cohen's kappa, confusion counts, per judge, the disagreements), to check a rubric before
labeling the rest. A label cannot be withdrawn, so judge labels the current votes do not back
(after a reset with `--max-rows`, or on undecided rows) are reported as `stale_judge_labels`,
and the triage stays `incomplete` until a run over every row replaces them. It is paid, so it
needs `--confirm`, but it writes nothing to Datadog, so `PUP_READ_ONLY` does not block it.
`--checks FILE.py` runs the user's row checks first (after `--rows` and `--max-rows`): a row
one fails is labeled `by: check` before any judge call and never judged, the preview runs them
too (one that raises is a `usage` error naming the row), and they are not frozen, so rows they
labeled are skipped without `--checks` (`skipped_checked`); `triage.json` records a council's
file as `checks_file`, which a dataset's step to finish the triage passes again while it
exists. With `--checks` alone it calls no
judge and labels at once, without `--confirm` (refusing `--judges`, `--concurrency` and
`--reset-triage`), reporting `stale_check_labels`; `dataset label --file` refuses `by: check`.
Tests replace the judges' clients (`cli/triage.py`'s `open_clients`) with a fake.

Judges can run on Anthropic or OpenAI too: `--judge-model`, `--eval-judge-model` and each
`--judges` entry take `anthropic/<model>` (`ANTHROPIC_API_KEY`, or the Anthropic SDK's own
login), `openai/<model>` (`OPENAI_API_KEY`), or otherwise a Fireworks model id such as
`accounts/fireworks/models/<model>` or its alias (`deepseek-v4p1-flash`, `glm-5p3-flash`:
`judge_models.JUDGE_SHORT_NAMES`), and a triage council can mix them. An alias resolves to the
full id, which previews print and `triage.json` and `evaluation.json` freeze. A judge on a
reasoning model asks for the lowest reasoning effort the model takes (`low` on Anthropic and
OpenAI; per model on Fireworks, `judges.FIREWORKS_REASONING`). Without `--judge-model`
(`--eval-judge-model`, `benchmark --judge-model`) or `--judges`, the judges are the ones the
directory froze (an evaluation with results, `triage.json`), else a default by the credentials
that resolve offline (environment variable, keychain, or the Anthropic SDK's login), for a
provider whose judge package is installed; an Anthropic key without the `anthropic` package is
passed over, and the source says so. The defaults are `judge_models.EVALUATE_DEFAULTS` and
`TRIAGE_DEFAULTS`, picked by `judges.default_judges`: `evaluate` takes
`anthropic/claude-haiku-5-5`, else `openai/gpt-5.4-mini`, else Fireworks'
`deepseek-v4p1-flash`; triage takes Haiku and `gpt-5.4-mini` with both keys, Haiku and
`anthropic/claude-sonnet-5-5` with Anthropic's, `gpt-5.4-mini` and `openai/gpt-6-astra` with
OpenAI's, else the two Fireworks models. Previews name the judges and where they came from
(`judge_source`, `judges_source`), triage's `next_command` pins `--judges` as evaluate's pins
`--judge-model`, and `doctor`'s `ready` map reads the default from its own key and package
checks. A preview whose judge or compare model has no key, or a malformed one, fails with
the key's login hint (`usage`, exit 2). The test conftest clears every provider key variable (except in `live`
tests), so the default tests see the Fireworks defaults. Anthropic judges need the `anthropic`
extra (`uv sync --all-extras`; `uv sync` is exact, so a bare `uv sync --extra anthropic`
removes the other extras):

```sh
uv run pup-finetune auth login anthropic       # or openai; hidden prompt, OS keychain
uv run pup-finetune auth status                # anthropic-sdk when the SDK signs in itself
uv run pup-finetune evaluate /tmp/run --judge-model anthropic/claude-opus-5-5
                                     # preview names the judge's provider and endpoint
```

Tests never call them: `tests/test_anthropic_client.py` drives the client with a fake SDK
client (and the real SDK over a mock `httpx2` transport), and the OpenAI request policy is
checked over `httpx.MockTransport`. `PUP_FINETUNE_LIVE=1` with `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` set runs `tests/test_judges_live.py`, one paid judge call per provider.

To see the models the providers can fine-tune (both are read-only, so they work with
`PUP_READ_ONLY` set):

```sh
uv run pup-finetune models list                               # every provider's profiles; no key
uv run pup-finetune models list --provider fireworks --check  # plus the tokenizer and a preflight
```

`--check` asks each model's provider about it (one read-only call per model, nothing paid) and
needs that provider's key: `BASETEN_API_KEY` or `FIREWORKS_API_KEY`, or the OS keychain after
`pup finetune auth login <PROVIDER>`.

Training on Baseten Loops is paid: `--provider baseten --model qwen3p5-9b` (or
`glm-5p3-flash`, `qwen3p8-27b` or `kimi-k3`) needs the `baseten` extra and a Baseten API key
(`BASETEN_API_KEY`, the OS keychain after `pup finetune auth login baseten`, or the default
remote's key in `~/.trussrc`, read only). Its preview asks Baseten, read only, whether the
workspace can train the model; `--confirm` starts a billed dedicated trainer, deactivated when
the session ends.
`result.json` records the best sampler checkpoint as `deployable_checkpoint`, and `deploy`
serves it as a dedicated Baseten model named after the deployment id (the run id made safe, or
`--deployment-id`), on an instance type with exactly the GPUs of `--shape TYPE:COUNT` (default
`H100:1`), scaling 0 to 1 replicas. The workspace needs BDN volumes and an `hf_access_token`
secret (the deployment downloads the base weights from Hugging Face). A name Baseten already
has is refused before anything is created. `undeploy` deletes that Baseten model with its
deployment; the checkpoint is kept, so the run can be deployed again.

Training on Fireworks is paid: it needs the API key and the Training SDK, which comes with the
`fireworks` extra (`uv sync --all-extras`).

```sh
uv run pup-finetune train --provider fireworks --model qwen3p8-27b --dataset /tmp/ds \
  --run-dir /tmp/fw-run --max-epochs 1   # preview only (exit 2); add --confirm to train
```

The best epoch is promoted to a Fireworks model (named after the run id; a new best found by
resuming a run that already promoted one gets the id with `-e<epoch>s<step>`) before the
session ends, and `result.json` records it as `promoted_model`. `deploy /tmp/fw-run`
previews an on-demand Fireworks deployment of it (paid per GPU-second while a replica runs,
until `undeploy`); `--confirm` creates it (first validated shape unless `--shape`, at most
one replica, scaling to zero when idle, hot loading off), waits until it is `READY`
(`--timeout-minutes`, default 30), smoke-tests one chat completion and prints the inference
model (`accounts/<account>/deployments/<id>`).
Training runs in the key's own account (Fireworks refuses a key that sees several accounts);
`FIREWORKS_ACCOUNT_ID` only overrides the account lookup, i.e. the account id pup-finetune
records.

Train only on synthetic rows, pushed to `pup-finetune-dev`, while developing. `deploy` creates
deployments through each provider's API, so an agent must pass `--confirm` only with the
user's explicit authorization. Always `undeploy` what you deploy while developing.
`--max-seq-len <n>` (the `max_seq_len` provider option, which `--provider-option
max_seq_len=<n>` still sets; at least 1024 on Baseten) cuts the profile's context to `n`
tokens: the run records the cut profile, and the preview already leaves longer rows out
(`exceeds_context`). The fake provider takes it too.

Add `--detach` to train in a background worker (output in `/tmp/run/worker.log`) and follow it
with `status /tmp/run --watch`. `--timeout DURATION` (`90s`, `10m`, `2h` or seconds; 90s by
default under a coding agent, i.e. `Runtime.under_agent`: `PUP_AGENT_MODE`, `AI_AGENT` or
`CLAUDECODE`) ends the watch with the status, `watch_timed_out: true` and the same watch as
`next_command`; Ctrl-C ends it the same way (exit 0), also when the process started with
SIGINT ignored, and never signals the training process. Under a coding agent, the watch a
result names as `next_command` pins `--timeout 90s`. `--provider-option step_delay_s=0.5
--batch-size 1` makes the fake slow enough to `cancel /tmp/run --confirm` (or press Ctrl-C in
the foreground), `--provider-option hold_file='"/tmp/hold"'` holds every step while that file
exists (tests use it to keep a run training deterministically), and
`--provider-option fail_at_step=9` simulates a crash to `resume /tmp/run --confirm` from. Use a
new `--run-dir` for each `train`, or none: without it a run goes in `./runs/<run id>` (a `runs/`
it creates gets a `.gitignore` holding `*`, and the result says so; `datasets/` gets one the
same way from `dataset pull` and `import`), every run command also takes the bare run id
(an existing path wins), `status` alone shows the newest run there and `runs` lists them. Run
that loop from a scratch directory, never the checkout, so no `runs/` lands in the repository.

Environment variables for development and unusual setups:

- `PUP_FINETUNE_PUP` - the pup executable to run (a path, or a name on `PATH`) instead of `pup`.
- `PUP_FINETUNE_KEEP_DD_TOKEN=1` - pass `DD_ACCESS_TOKEN` through to the `pup` processes we
  start. By default it is removed, so each child `pup` loads and refreshes its own stored OAuth
  token instead of reusing the short-lived token pup handed us. Set it only if you
  authenticate pup through environment variables alone.
- `PUP_FINETUNE_DEV=1` - development mode: enables dev-only providers such as the offline
  `fake` provider (`--provider fake`). `1`, `true`, `yes` and `on` (any case) turn it on;
  anything else is off, with a warning for a value that is neither on nor off.
- `PUP_FINETUNE_QUIET=1` - no closing summary on stderr after the result (the warnings are
  still printed). On values as for `PUP_FINETUNE_DEV`.
- `PUP_FINETUNE_PROGRESS=steps` - a progress line per training step and per evaluated pair
  even off a terminal, where they are otherwise throttled (`auto`, the default).
- `PUP_FINETUNE_BASETEN_CLI` - the `baseten` CLI `train --remote` runs (a path, or a name on
  `PATH`) instead of `baseten`.

The `datadog-finetune` agent skill (`skills/datadog-finetune/SKILL.md`, the one source, in the
layout the skills CLI discovers) ships in the wheel as `pup_finetune/_skills/` through
`[tool.hatch.build.targets.wheel.force-include]`; an editable install reads the repository's
copy. `skill show` prints it; `skill install` runs `npx --yes skills add <bundled dir> --skill
datadog-finetune` (passing on `-a`, `-g` and `-y`, its stdout sent to stderr, and its
environment without the Datadog secrets or any provider key variable, since npx downloads and
runs the skills CLI from npm) when `npx` is on `PATH` and `--dir` is not given, and otherwise
copies it to `.agents/skills/` and `.claude/skills/` under the project (or the home directory
with `-g`). Neither makes a Datadog or provider call, so both work with `PUP_READ_ONLY` set.
When you change a command the skill names, keep the skill in step:
`tests/test_doc_commands.py` checks every `pup finetune` command and flag in it (and in the
README, `docs/train.md` and `docs/evaluate.md`) against the app, and `tests/test_skill.py`
checks its structure (the flow and start table, eight short rules, the preview and status
templates, Steps 0 to 7 with a `Check:` list each, and the error-code and debugging tables). Its
commands use example paths and run ids (`./ds`, `ft-...`), never shell variables, and
`<UPPER>` only for values to fill in.
`SKILL.md` has no license header: its frontmatter must come first, and
`scripts/check_headers.py` checks only code.

Opt-in test suites: `PUP_FINETUNE_LIVE=1` runs `live` tests (real external services);
`PUP_FINETUNE_TOKENIZER_TESTS=1` runs `tokenizer` tests (downloads real tokenizers, needs the
`hf` extra): `PUP_FINETUNE_TOKENIZER_TESTS=1 uv run pytest -m tokenizer`. The live Baseten
deploy test also needs `PUP_FINETUNE_BASETEN_CHECKPOINT`, a `bt://loops:` sampler checkpoint
of `qwen3p5-9b`: it deploys it on a billed H100 and always undeploys it.

## Layout

Source lives under `src/pup_finetune/`.

- `cli/` - the Typer app; one module per command group. `_views.py` has what `train`,
  `resume`, `status` and `evaluate` print (`train_view`, `status_view`, `evaluation_view`):
  the decision in reading order and a `files` section naming the files that keep the rest
  (`result.json`, `summary.json`, ...; the files themselves are unchanged), with keys that
  are sometimes `null` kept and only whole sections left out; and `TABLE_VIEWS`, the jq
  expression each list-shaped command (`models list`, `auth status`, `doctor`, `dataset
  sample`, `runs`) passes `pup format` for `--output table` or `csv` without a `--jq`. `_summary.py` has the closing
  summary of each command (`SUMMARIES`, by command path), generated from the result mapping
  alone; `run_command` prints it, after the result's warnings, unless `PUP_FINETUNE_QUIET`
  is on. `dataset.py` has `dataset import`,
  `fetch`, `pull`, `show`, `sample`, `label` and `push`, and `dataset_next`, the next
  command and steps a dataset directory's state calls for (unlabeled, or labeled only `by:
  check` with rows left: `dataset sample`; labeled: training steps, since the model is the
  user's to choose; the provider follows from it and the run directory defaults); `train.py`
  has `train` (`provider_for_model` infers `--provider` from any of a model's names in the
  registry, failing on a name several providers offer; the rerun pins the alias),
  `resume`,
  `status`, `cancel` and the hidden `_worker` a detached run executes (and `train
  --evaluate`), and `run_next`, the next command a run's phase calls for; `models.py` has `models list` (every provider's
  profiles, or `--provider`'s; with `--check`, the renderer and tokenizer and a preflight per
  profile);
  `runs.py` has `runs`, the runs under `./runs` (or `--dir`) newest first, over
  `workspace.list_runs`;
  `evaluate.py` has `evaluate`, which only parses its flags and emits what
  `evaluation/runner.py` returns; `benchmark.py` has `benchmark`, which does the same for
  `evaluation/benchmark.py` (a `--models` entry may be `PROVIDER/MODEL`, which the rerun
  writes for a model several providers offer); `deploy.py` has `deploy` and `undeploy` (over
  `deployment.py`); `triage.py` has `dataset triage` (over `curation/triage.py`; its
  `open_clients` builds one chat client per judge provider); `skill.py` has `skill show` and `skill
  install` (its `find_npx` and `run_npx` are what tests replace); `help.py` has `help
  [COMMAND...]`, the text `<COMMAND...> --help` prints (pup intercepts `--help` after an
  extension under agents, but not `help`), at the same width, and points
  Typer's usage-error "Try ..." line to it. `_common.py` names the program `pup finetune`
  under pup (`PUP_OUTPUT` set) and `pup-finetune` standalone (`prog_name`), passes the
  running command's path to `output.emit`, and prints `Runtime.warnings` once per command.
  `_flags.py` has `require_enablers`, which refuses an option given without the one it needs
  (each command keeps the table next to its options, e.g. `train.TRAIN_NEEDS`: every
  `--eval-*` needs `--evaluate`), and `require_alone` (`dataset triage --print-template`
  takes nothing else), and `renamed`, which reads a renamed option and takes its old name, a
  hidden option, with a warning (`--max-cases-per-trajectory`, `--eval-max-cases-per-trajectory`,
  `dataset sample --n`; remove them in the next release). `_dirs.py` names the wrong kind of
  directory: a dataset given to a run command, a run given to a dataset command or as `train
  --dataset`, a dataset as `train --run-dir`; its `resolve_run_dir`, which every run command
  calls first, reads a bare run id as `runs/<id>` and says so on stderr. `_presentation.py` is how commands show in
  `--help`: every command has a `CommandDoc` (a short help of 12 words or fewer naming its
  gate, a description of at most three sentences, the rest after `\f` for the generated docs,
  examples and a `Safety` record, which renders as the help's last line and as the safety
  table of `docs/commands.md`), `COMMAND_ORDER` lists commands in workflow order (for `--help`
  and the docs), and `Command` and `Group` are the Click classes everything is built with
  (usage without `{DIR}` braces, no ranges in the type column, `--help` in an option panel,
  every missing required parameter named at once, and Click's errors printed by
  `_common.report_click_error` as `Error [usage]: ...` with a hint naming `help <command>`,
  never in Typer's box);
  `default_note` writes a default the option itself cannot hold. `tests/cli/test_help_layout.py`
  holds every `--help` to 80 columns with nothing cut off and at most 60 lines (`train`, with
  37 options in six panels, 65), and parses every example against the app.
- `skill.py` - the bundled agent skill: `bundled_skills_dir` (the package's `_skills/`, else
  the repository's `skills/`), `npx_argv`, `npx_env` (the environment without
  `datadog/pup.py`'s `SECRET_ENV` and every registered provider key variable) and
  `install_files` (the fallback copy; a different copy is replaced only when asked).
- `deployment.py` - what `deploy` and `undeploy` run, provider-neutral: `deploy_target` (a
  finished run's promoted model, else its deployable checkpoint), `deployment_id_for` (the
  run id made safe, or `--deployment-id`), `execute_deploy` (writes `deployment.json` as
  `REQUESTED`, with the deployment's resource name and what the create sends (`request`),
  from the provider's `on_creating` callback right before the create, removed again if the
  provider refused; then
  from the provider's `on_created` callback, before the wait, on `READY` and with the smoke
  test; a failure once it may exist, a smoke test that raises included, is recorded with an
  undeploy hint; one deployment per run)
  and `execute_undeploy` (idempotent; records `deleted_at`; the model is kept; a `REQUESTED`
  receipt's deployment is read back with `Provider.owns_deployment` and deleted only when it
  is the one requested). The receipt records the provider's `Endpoint`: `url` (the
  OpenAI-compatible base URL), `inference_model` and `api_key_env` (the key's environment
  variable, never the key), from which `deployment_usage` builds the deploy result's `usage`
  (with an example request); the result's `next_command` is `null` and `undeploy_command`
  says how to delete it.
- `runtime.py` - non-secret settings from the environment (`PUP_OUTPUT`, `PUP_READ_ONLY`, ...),
  and `under_agent`, whether a coding agent drives the process (`PUP_AGENT_MODE`, `AI_AGENT`
  or `CLAUDECODE`), for defaults that suit an agent's bounded tool calls (`status --watch`'s
  timeout). `PUP_READ_ONLY` is read-only unless it is `0`, `false`,
  `no`, `off` or empty (any case); an unrecognized value is read-only too, with a warning in
  `warnings`.
- `errors.py` - expected failures, their exit codes and their stable `error_code`s
  (`ERROR_CODES`, each with its exit code and meaning, which `docs/commands.md` lists; the
  error line reads `Error [<code>]: ...`). `DirectoryUsageError` and its subclasses
  (`NotADatasetDir`, `NotARunDir`, `RunDirNotEmpty`, `SettingsConflict`, `WrongPhase`) are
  both a `UsageError` and a `StateError`, so code that caught either still does, and exit 2;
  `Interrupted` is a Ctrl-C or SIGTERM. The `--confirm` and read-only guards
  (`require_confirm(..., read_only=...)` adds `read_only` to a preview and offers no
  `--confirm`), `READ_ONLY_HINT`, and `install_hint` / `reinstall_hint`, the one wording of
  how to install an extra (`uv sync --all-extras`, never a bare `uv sync --extra X`).
  `tests/test_hint_commands.py` parses every backticked `pup finetune ...` in a string in
  `src/` with the app and holds every backticked `uv ...` to an allowlist.
- `next_steps.py` - the `next_command` rule (see Conventions): `command` (every word quoted),
  `template` (a `next_steps` command, `<UPPER_CASE>` placeholders left as they are), `step`
  and `problems`, which checks a result against the rule (the test conftest runs it on every
  emitted result; tests that break it on purpose are marked `any_next_command`).
- `display.py` - how values read in results and hints: `display_path` (relative to the
  current directory when inside it, else absolute; `next_steps.command` and the hints'
  `quoted`, the shell-quoted `word`,
  use it for path words), `significant` (4 significant digits) and `rounded_floats`, which
  rounds only measurements (`MEASUREMENTS`: losses, rates, deltas, CI bounds, times, spend
  estimates, ages) and never a value the user set (`UNROUNDED`: `settings` and the like),
  and `joined`, the one way a list reads in a sentence (`a, b and c`).
- `output.py` - the one stdout result (redacted, measurements rounded with `display.py`, keys in
  the order they were built, text not escaped, plain lists and objects on one line when they
  fit; through `pup format` under pup, with `--command
  'finetune <command>'` and the result's `next_command` as `--next-action` for pup's envelope
  metadata, and a command's table view as `--jq`), progress on stderr (`info`, stamped `HH:MM:SS` off a terminal, `worker.log`
  included), `echo_warnings` (`warning: <text>` for every warning a result holds, read only
  from the result's own mappings, `WARNING_PATHS`, never from trace data in it; each once)
  and `summary` (a few lines on a terminal, one off it).
- `progress.py` - `TrainingProgress` (the loop's `on_event`) and `EvaluationProgress` (the
  replay's `on_progress`): on a terminal, or with `PUP_FINETUNE_PROGRESS=steps`, a line per
  step and per pair; otherwise a step line at most every 30 s or quarter epoch, each epoch's
  checkpoint folded into its epoch line, and an evaluation line every 10% of the pairs
  (counting failures cut off at `--max-tokens`, unparseable or without a verdict) plus one per
  error and one for the first failure of each cause. A failed pair's line names its cause
  (`failure_cause`: the cut-off with the `--max-tokens` hint, or a compare model's own limit,
  `format_valid` false, the judge's reason on one line, the no-verdict flaw or the error's
  message). Retries inside the judge and sampler clients print no evaluation line yet
  (training's retries do). Tests pass `write`, `clock` and the modes in.
- `storage.py` - atomic writes, `0600` files and `0700` directories, directory locks, frozen
  settings, and `identity_changes`, the keys two recorded identities differ in.
- `doctor.py` - what a `doctor` check reports (`CheckResult`: `status`, `detail`, `label`,
  `needed_for`, `fix`, `required`) and the registry of extra checks modules add with
  `register_check`, which `doctor` runs after its own.
- `readiness.py` - `doctor`'s provider and judge checks, built from the registries: each
  provider's key (`CredentialSpec.needed_for`) and packages (`ProviderInfo.packages`, found
  with `find_spec`, never imported), the judge-only keys and the Anthropic SDK, the live
  `preflight_checks` for `--provider NAME --check`, and `ready_map` (each step, ready or
  `blocked_by` which checks, `read_only` included). It imports `judges` only inside its
  functions (for `JUDGE_PACKAGES` and `pick_default`).
- `credentials.py` - provider API keys (never Datadog's): a `CredentialSpec` registry, resolution
  from the environment variable, then the OS keychain (`keyring`), as `SecretStr`; `store` refuses
  when there is no OS keychain rather than writing a file. `auth login/status/logout` use it.
  `key_source` is where a key would come from without reading it out, or a `KeyUnreadable`
  when the lookup raised anything; `doctor`'s key checks and the default judges both use it.
- `redaction.py` - masks resolved secrets and `Bearer` tokens in error output and logs.
- `user_checks.py` - the user's Python checks, shared by `evaluate --evaluators` and `dataset
  triage --checks`:
  `load_checks` runs a file as a module named by its sha256 (`CheckModule`: its public
  functions in definition order, `function` with an arity check), `call` runs one check on
  deep copies under one lock and reads `True`, `False`, `None` or `(bool, reason)`; anything
  else, or an exception, is a `CheckError` (not a `FinetuneError`), redacted.
- `concurrency.py` - `DaemonPool`, worker threads for parallel remote calls that never hold
  up the process's exit, so Ctrl-C stops at once instead of waiting for a call in flight.
- `http_common.py` - what the `httpx` clients that send a provider key share: `bearer_key`
  (refuses an empty or malformed key and registers it for redaction), `CLIENT_SOURCE`
  (`pup-finetune/<version>`) and short, redacted failure text.
- `inference.py` - `ChatClient`, an OpenAI-compatible `POST {base_url}/chat/completions` client
  over `httpx` (no SDK): bounded retries with backoff and jitter on 408, 429, 5xx and connection
  errors (honoring a capped `Retry-After`), no retry on other 4xx, 401/403 as
  `ProviderAuthError` with a login hint, 429 after retries as `ProviderCapacityError`, the rest
  as `TransientError` or `ExternalError`; `ChatCompleter` is its protocol (with an optional
  `response_schema`). `RequestPolicy` is what an endpoint takes (`OPENAI_POLICY`:
  `max_completion_tokens`, no temperature, JSON mode); a `reasoning_effort` is sent as given.
  A request longer than the model's context window is a `ContextLimitError` (never retried;
  `is_context_limit` reads each provider's error answer). `JudgeConfig` is the judge's
  provider, model, endpoint, credential, sampling settings and reasoning effort (`identity()`
  for resume checks, `spec` for the command line).
- `judge_models.py` - judge model names with no imports, so help texts can use them: the
  default judges by credential (`EVALUATE_DEFAULTS`, `TRIAGE_DEFAULTS`: ordered
  `JudgeDefault` rows of the credentials a row needs and its judges; `BY_KEY` for help) and
  `JUDGE_SHORT_NAMES`, the short names of Fireworks judge models.
- `judges.py` - judges on any provider, no SDK import: `parse_judge` reads a spec by prefix
  (`anthropic/` or `openai/`; anything else is a Fireworks model id, an alias resolved) with the
  provider's defaults (`PROVIDERS`; the lowest reasoning effort each model takes,
  `FIREWORKS_REASONING` per Fireworks model), `default_judges` (and `pick_default`) the
  default judges by credential with where they came from, `JUDGE_PACKAGES` the package a
  judge provider needs (what `doctor` checks, a default requires and `check_available`
  refuses a judge without, missing or broken, up front), `require_key` a missing key (offline:
  the environment, the keychain, or the Anthropic SDK's own login; the fake judge and compare
  model need none), which the `evaluate`, `train --evaluate`, `benchmark` and `dataset triage`
  previews call for every model they would call, so they never offer a `--confirm` that would
  fail on it (a read-only `evaluate`, `train --evaluate` or `benchmark` preview, which offers
  none, warns instead of a missing key: `warnings`; an unreadable keychain still fails), and
  `open_client` and `open_clients` (one per provider) build the clients. A run opens them up
  front, before anything is generated, so a malformed key, a missing SDK or a locked keychain
  fails before anything is paid, but only those its preview checked: the judge when the same
  count the preview showed (`EvalPlan.work`, `BenchmarkPlan.judge_calls`) has a judge call
  left, so a complete evaluation or benchmark rerun opens none, and in a triage council the
  judges with votes pending (`TriagePlan.pending_judges`).
- `anthropic_client.py` - `AnthropicChatClient`, the Anthropic judge (and compare model)
  client over the official SDK (the `anthropic` extra), the only module that imports
  `anthropic` (and its `httpx2`): the messages as `anthropic_messages.py` converts them,
  OpenAI tools as Anthropic tools, `tool_use` answers back as OpenAI `tool_calls`, no sampling
  parameters, `output_config` format
  and effort (the judge's), "prompt is too long" as `ContextLimitError`, `fallbacks="default"` through the beta API on the models that have it (never for a compare model: `fallbacks=False`), `stop_reason`
  mapped to `finish_reason`, typed SDK errors mapped to ours. Tests pass a fake SDK client.
- `anthropic_messages.py` - the one OpenAI-to-Anthropic message conversion, with no SDK
  import: every system message joined into `system`, assistant `tool_calls` as `tool_use`
  blocks, `tool` messages as `tool_result` blocks of user turns, empty messages dropped and
  consecutive turns of one role merged. `anthropic_client.py` sends it, and the compare skip
  rule checks it.
- `judge_credentials.py` - registers the `anthropic` and `openai` `CredentialSpec`s (imported
  by the providers package); `anthropic_sdk_login` detects the Anthropic SDK's own login,
  reported as the source `anthropic-sdk`.
- `datadog/` - Datadog access via `pup`. `pup.py` is the client every Datadog call goes
  through: `pup --no-agent --output json ...`, a scrubbed environment, timeouts, and failures as
  `ExternalError`. `spans.py` exports LLM spans through the Export API
  (`POST v2/llm-obs/v1/spans/events/search` via `Pup.api`): `SpanQuery` (the filter, plus
  `settings()` to freeze), `iter_spans` and `iter_span_pages` (cursor pagination, retries of
  timed-out pages, an optional raw page cache that resumes without refetching, `ExportStats`),
  and tolerant `LLMObs*` models that skip and count spans they cannot read (`UnreadableSpan`).
  `datasets.py` has `LLMObsDatasets`: projects (`pup llm-obs projects list/create`), datasets
  (listed and created through `Pup.api` on `v2/llm-obs/v1/<project>/datasets`, since pup
  1.10.7 cannot read the response of `pup llm-obs datasets create`), `records-add --confirm`
  and `records-all`; lists are read page by page, a create whose response is lost is found
  again by name. `jsonapi.py` has the JSON:API helpers both clients share: the media type,
  `private_json_file` (`--file` bodies are `0600` in a private temporary directory, removed
  afterwards) and `object_id`. `experiments.py` publishes an evaluation: `LLMObsExperiments` (`pup llm-obs
  experiments create --file`, `list`, `events list`, and spans plus metrics through `Pup.api`
  on `v2/llm-obs/v1/experiments/<id>/events`) and `ExperimentsPublisher`, the evaluation's
  `Publisher`: one experiment per target on the pushed dataset version, a `replay_trajectory`
  root span per test row (`dataset_record_id` from `push.json`) and a `replay_assistant`
  child per case, metrics stamped at submission, at most 100 spans a request, uuid5 span ids,
  `experiments.json` and `published.json` so a rerun sends nothing twice (a batch whose
  response was lost is sent again with the same span ids), cancellation between requests,
  and a bounded read-back until the events are indexed.
- `data/` - training data. `schema.py` is the canonical row every provider consumes (OpenAI
  chat format plus its source) and the stable `RowErrorCode` reasons a row is rejected for;
  `jsonl.py` reads and writes rows and converts them to the provider form; `exclusions.py`
  records why a row was left out (`excluded.jsonl`), with `ExclusionCode`s for reasons the
  schema cannot see (conversion problems, dataset-level rules);
  `split.py` assigns deterministic, persisted train/validation/test splits per trace, one per
  group of traces asking copies of one request (`request_key`: the user messages lowercased,
  numbers as `0`, whitespace collapsed, hashed, `None` without user text; `split_groups`, a
  union-find that joins every trace asking one request unless the request is generic, asked
  in more traces than `generic_limit` (5% of the traces rounded up, and at least 5), which
  joins nothing and is reported as `generic`; the summary warns about a group of 3 traces or
  more holding more than 10% of the rows in a dataset of 50 traces or more
  (`SplitGroups.chained`); `assign_splits`, which never moves a recorded trace;
  `split_conflicts`, every group whose kept rows are in several splits, which the summary
  counts);
  `dataset_dir.py` is the dataset directory (`settings.json`, `rows.jsonl`, `excluded.jsonl`,
  `splits.json`, `summary.json`, `split_conflicts.jsonl` (drop labels for the kept copies
  outside their group's split, written with the summary while there are any), `raw/`, a
  pull's page cache, and `push.json`, the receipt `dataset push` writes) behind `dataset
  import`, `dataset pull` and `dataset show`;
  `labels.py` has the keep/drop `Label` and the `labels.jsonl` format (append-only, last label
  per id wins, a reason required to drop; `DatasetDir.kept_rows` leaves dropped rows out of
  training and split exports).
  `convert.py` maps one LLM span to a canonical conversation (`span_conversation`: input then
  output messages, a plain `value` as one message, tool calls and results, tools, `SourceRef`;
  reasoning omitted or preserved), with conversion issues as stable codes; `trajectories.py` has
  `rows_from_spans`, which dedupes spans by id, groups them by trace, collapses conversations
  that are a prefix or duplicate of another (so sub-agents stay their own rows) and yields a
  validated row or an `Exclusion` per kept conversation. `checks.py` has the span-level checks
  that run before conversion, in order, first failure wins (`span_error`; errored calls are kept
  with `--include-errors`); there is no `truncated_content` check, as the Export API gives no
  explicit truncation signal. `exclusions.py` also explains every code (`explain`). `pull.py` is
  what `dataset pull` runs (into a dataset, through its `raw/` cache, with the settings frozen
  first; tests run `collect` in memory), a relative time range resolved once (a re-run
  reuses the frozen window, `resolve_window`): export a page at a time, check, convert, drop
  repeated conversations, and stop at `--target-count` rows or `--max-candidates` spans before
  fetching another page. `--max-rows-per-trace` excludes a trace's rows past the cap as
  `trace_row_cap` (in the dataset too: `trace_capped`); `--sample random` reads every
  candidate, then deals rows round-robin over traces in a seeded order, stored rows first
  (`_ordered`); both are frozen only when set, so older `settings.json` files still match.
  Unreadable items are excluded as `unreadable_span` and count toward
  `--max-candidates` (`SpanQuery.max_spans` caps export items, readable or not); a re-run that reads
  further retires a stored row its trace now shows to be a prefix of a longer conversation
  (`retirements`, `contained_conversation`), and one with lower limits does not add such a row
  back (`contained`); both compare tools and messages as conversion collapses them
  (`comparable_tools` and `comparable_messages` in `trajectories.py`). With `--reasoning preserve`, a stored row a
  re-run reads more reasoning for is updated in place, never losing reasoning it has
  (`reasoning_updates`, `DatasetDir.add(updated=...)`), and a stored row that update makes
  repeat another stored row is retired as `duplicate_conversation` (`repeat_retirements`).
  `fetch.py` is `dataset fetch`: an Agent Observability dataset's current records back into
  rows (input messages plus expected output, the split from metadata or tags), frozen per
  dataset version, with the dataset's `push.json` so evaluations publish onto it.
  `push.py` is `dataset push`: `dataset_record`
  (input = the messages before the final assistant message plus tools, expected output = that
  message, metadata = row id, split and source, tags `split:*` and `source:pup-finetune`),
  `plan_push` (what would be sent, from local files only) and `push_dataset`, which sends only
  kept rows in batches of 100 and saves `push.json` (project and dataset ids, version, the
  record id per row, the batch `in_flight`) around every batch, so a re-push sends only new
  rows and an interrupted push resumes, first adopting the records of a batch whose response
  was lost (matched on `metadata.row_id` through `records-all`) so no row is added twice (a
  listing that is truncated or hit pup's page cap, `RecordsListTruncated`, or that holds
  items it cannot read, `RecordsListing.unreadable`, keeps the batch `in_flight` and stops;
  so do rows of it not found within `IN_FLIGHT_SETTLE`, 10 minutes, of `in_flight_at`, since
  a timed-out write may still land); rows dropped after a push are reported as
  `stale_records`, not deleted, and kept rows whose record no longer matches the digest
  `push.json` saved for it (`record_digests`; none in older receipts) as `changed_records`,
  not updated: a push only adds records (`compare_receipt` sorts the kept rows into pending,
  changed and stale, for `plan_push` and `pipeline_state` both).
- `curation/` - deciding which accepted rows to train on. `labels.py` has `apply_labels`
  (every id an accepted row, last label per id wins, repeats skipped, appended through
  `DatasetDir.append_labels`); `samples.py` picks a seeded, readable sample of rows for
  review (`row_summary` is a row's brief form; `review_markdown` is `sample --markdown`'s
  file, message text fenced); `pipeline.py`
  has `pipeline_state`, the `pipeline` block of every dataset result. `triage.py` is the judge council behind `dataset triage`: `Criteria` (rubric and/or
  rules, hashed), `build_messages` (the untrusted-data system prompt with the rubric, the
  whole row JSON-encoded in blocks), `parse_vote` (strict `{"keep", "reason"}`), `Vote`s in
  `judgments.jsonl` (counted only for the row content, rubric and judge they were cast for),
  `decide` (strict majority keeps, a tie drops, a missing vote is undecided, a `too_long`
  vote drops), `plan_triage` (rows of `only` not labeled by the agent or a person, or with
  `agreement` only those, in a fixed pseudo-random order for `--max-rows`; settings frozen in
  `triage.json` once votes exist) and `run_triage` (a thread pool, votes appended as they
  arrive, failed calls reported and retried next run, a row's calls skipped once a judge
  found it too long, labels `by: judge` for decided rows, judge labels the votes do not back
  reported as stale, or with `agreement` no label and `measure_agreement`'s `Agreement`; a
  torn last vote line is cut before appending), with `checks` `run_row_checks` (the rows a
  check fails, `CheckFailure`s, labeled `by: check` first, also by `apply_check_labels` for
  checks alone, and with `agreement` compared apart, `check_agreement`). `rubric_template.md`
  ships in the wheel. Curation builds on
  `data/`; `data/` never imports `curation/`.
- `training/` - the shared, provider-neutral SFT loop. `datums.py` renders rows into
  `TokenizedExample`s (per turn when a renderer is not prefix-stable) and rejects rows that
  exceed the context or train on nothing; `batching.py` shuffles rows per epoch and packs
  optimizer steps into microbatches; `budget.py` caps trained tokens and wall-clock spend;
  `loop.py` has `run_training` (pipelined steps, validation loss, early stopping, best epoch,
  epoch-end and time-based checkpoints, cancellation, deterministic resume at an epoch and
  step, and the `finalize` hook, whose failure raises `FinalizeFailed` carrying the result;
  it sends each step its scheduled learning rate, `TrainSpec.learning_rate_at` over
  `max_steps`, so a resume sends the same rates);
  `records.py` has its records (`EpochMetrics`, `TrainResult`, `ResumePoint`, `Resume`);
  `promotion.py` has `Promotion`, the `finalize` hook that promotes the best checkpoint
  before the session ends (re-creating its sampler checkpoint when that session expired) or
  records it as the deployable checkpoint, recording each promotion request per checkpoint
  first (a new best after an earlier promotion gets its own model id,
  `model_id_for_checkpoint`, so a model id never holds two checkpoints); `runner.py` has
  `start_training` and `resume_training`, which open a session for a run directory and train
  or resume in it (`start_training` takes an `after_finalize` hook, run in the session after
  a successful promotion with the run's spend guard; its failure only adds a warning);
  `inputs.py` renders a dataset's train and validation rows for a run (and checks, for an
  existing run, that the dataset and renderer did not change); `worker.py` runs a mode
  (`train`, `resume`, `finalize`) in the foreground or a detached worker (`spawn_worker`:
  `python -P -m pup_finetune _worker` in a new session, in the run directory, output in
  `worker.log`, the environment inherited and no secret passed on; the worker notes how it
  ended in `worker.json`), and has the liveness checks behind `status` and `cancel`.
  `remote.py` is `train --remote`, provider-neutral: `stage_run` (pup-finetune's source and
  `uv.lock` from the checkout, the kept train and validation rows with their splits, and the
  run files, in a private temporary directory), `submit_remote` (`remote.json` first, the
  files rolled back when the provider refused before creating anything) and `sync_remote`
  (the job's state and state reports, a `WorkerStatus` stand-in, adoption of a job a lost
  answer did not name); `report.py` is the return channel: `StateReporter` (`_worker
  --report-state` prints checksummed `PUPFT1` snapshots of the run files on stdout),
  `newest_snapshot` and `apply_snapshot`.
- `workspace.py` - the working directory's `runs/` and `datasets/`: `default_run_dir`
  (`runs/<run id>`), `find_run_dir` (an existing path, else `runs/<id>` for a bare run id),
  `run_dirs`, `latest_run_dir` and `list_runs` (newest first by `created_at`), and
  `ignore_new_parent`, which creates a missing `runs/` or `datasets/` atomically with a
  `.gitignore` holding `*` (a temporary directory renamed into place) and returns a
  `ParentSetup`: the note the result shows in `notes` (echoed on stderr as `note: ...`), or
  a warning when an existing one lies unignored in a git work tree.
- `runs.py` - the run directory (`run.json`, `state.json`, `epochs.json`, `result.json`,
  `deployment.json`, `remote.json` (the job of a `train --remote` run), and `evaluation.json`, each evaluation of the run by its directory with
  every setting, how it stands and its headline, written by
  `evaluation/runner.py`'s `record_evaluation`; `status` names the pinned command that
  continues one, or `deploy` once one is complete, unless the missing push kept it
  unpublished (its `publication` is `skipped`): then the steps are built from the run as it
  is now, by the same `_push_first` as after the evaluation, less its compare suggestion
  (the dataset's current `push.json`, the evaluation's pinned command, `deploy` unless the
  run is deployed, paths from the current directory), with no next command while a push is
  needed or the receipt cannot be read, and once one was made the evaluation's command,
  whose preview publishes it),
  `RunState`, which the loop reports into, and `execute_run`, which holds the directory lock
  while training and always writes `result.json`, also on Ctrl-C or SIGTERM (reported as
  `TrainingTerminated`, a `FinetuneError`). Any error during training leaves the run
  `interrupted`, resumable from its last training checkpoint; a failed promotion leaves it
  `failed` with `resume_mode` `finalize_only`, so a resume only retries the promotion. Every
  error that leaves the run `interrupted` or `failed`, a `FinetuneError` included, ends its
  hint with `Next: pup finetune resume RUN_DIR` (`resume_hint`).
  `result.json` records `promoted_model`, `deployable_checkpoint` and `finalize_error`. With
  `signals="cancel"` (what the CLI uses), SIGTERM and the first Ctrl-C cancel gracefully
  instead: stop at a step boundary, checkpoint, promote the best epoch, end `cancelled`; with
  `heartbeat_s`, `state.json` is also saved on a timer so its heartbeat stays fresh.
- `rendering/` - chat rows to model tokens and completions back to messages. `tokens.py` has
  `TokenizedExample`, the provider-neutral, next-token-shifted token and weight arrays;
  `base.py` has the `Renderer` protocol and `RendererError` (`prompt_refusal` says, loading
  no tokenizer, which prompts `generation_prompt_tokens` refuses). `hf.py` has
  `HFChatTemplateRenderer`, the one renderer every provider shares: the model's Hugging Face
  chat template and tokenizer at a pinned revision, one example per trainable assistant turn
  (the prompt tokenized whole, as the template tokenizes it: its text encoded whole, or
  `ChatTokenizer.render_tokens` when the tokenizer's own code tokenizes the chat (`None`
  otherwise); then the completion through its stop token, re-encoded after the prompt when a
  token merges across the boundary, which a tokenizer whose `render_tokens` is not `None`
  refuses instead; two template renders per turn), and a per-row-checked one-example fast
  path for families that pin `prefix_stable` (every trainable turn a token prefix of the whole
  render). It also has the narrow `ChatTokenizer` seam (tests inject
  `tests/fakes/tokenizer.py`), the lazily imported `transformers` adapter (the `hf` extra;
  `load_tokenizer` passes `trust_remote_code` only for a family that sets it, always with the
  pinned revision as `revision` and `code_revision`, refuses a tokenizer class from any other
  repository or commit, and a remote tokenizer's identity is its pinned `model@revision`),
  and `ChatEncoder` and `EncodedChatTokenizer`, for a tokenizer that ships no chat template.
  `encoders.py` has those encoders (`DeepSeekV4Encoder`, DeepSeek-V4's
  `encoding/encoding_dsv4.py` for the messages we train on, held to it by the `tokenizer`
  tests). `parsers.py` has the `ResponseParser`s (`Qwen3XmlToolParser`,
  `Qwen3_5XmlToolParser`, `GlmXmlToolParser`, `JsonToolCallParser`, `DeepSeekDsmlParser`,
  `KimiXtmlParser`, `MuseAtemParser`) that turn sampled text back into a message (a
  `ControlTokenParser`, Kimi's, is also given the completion split at its control tokens, so
  text that spells one stays text: `Renderer.parse_completion`, which the samplers call);
  `families.py` is the model-family table (`qwen3_8`, `qwen3_5`, `glm5_3`, `kimi_k3`,
  `deepseek_v4`, `muse_glimmer`), each entry a tokenizer, revision, parser and stop tokens,
  and when it needs them `trust_remote_code` (Kimi K3 only), a `chat_encoder` (DeepSeek-V4)
  a `rewrite`, a `MessageRewrite` that turns each message into the messages the template
  renders faithfully (Muse Glimmer writes text beside tool calls as its own message to the
  user; its template's date is pinned in `chat_template_kwargs`), and a `check`, a
  `PromptCheck` that refuses a prompt the template opens no assistant turn after, and a
  trained turn after it, with `unsupported_message` (DeepSeek-V4 needs a user or tool message
  before each turn); each by name and version in the renderer identity, which leaves out
  what a family does not have.
  The `hf` extra includes `tiktoken`, which Kimi K3's tokenizer code imports.
- `evaluation/` - replay evaluation, provider-neutral. `cases.py` turns a dataset's kept
  test-split rows into `ReplayCase`s (`cases_from_dataset`, `build_cases`): one per trained
  (`weight` not `0`) assistant message with a visible action, with the recorded prefix and
  tools as input and the recorded message as the reference, capped per trajectory
  (`max_cases_per_trajectory`) with evenly spaced, seeded picks that always keep the final
  turn. `exclusions` names the cases a model cannot replay, each with a stable `code`: a turn
  with no message before it (`empty_prompt`, for every provider) and a prompt the model's
  renderer refuses (`prompt_refusal`, the renderer's code). Tools are never executed, so
  replay measures agreement with the recorded behavior, not task success. `metrics.py` has
  the deterministic tool-call metrics (`MetricName`, `score_case`, `aggregate`; `None` when a
  metric does not apply) and `schema_errors`, a
  validator for the common JSON Schema subset tool declarations use. `judge.py` is the LLM
  judge: `judge` asks whether a candidate action is an acceptable substitute for the reference
  (the recorded data JSON-encoded in tagged blocks marked untrusted, reasoning never shown) and
  parses `{"pass", "reason"}` strictly (`parse_verdict`, outside any `<think>` block; anything
  else fails as `unparseable`, or `truncated` when the judge hit `max_tokens`, and a refused
  answer as `refused`); a malformed candidate fails without a call, and a judge outage (or a
  case longer than its context window) raises rather than failing the case (replay records
  it as the pair's error; the result's hint says a retry cannot fix a case too long). `calibrate` checks
  that references pass and both of their `corruptions` fail on a few mixed cases; `check_judge_differs` refuses a judge that is the model being trained or the run's
  tuned model (`model_id_for_run`, and the promoted model once there is one).
  `fake_judge.py` is `FakeJudge`, an offline, deterministic judge for development mode and
  tests (same tools with consistent arguments, or mostly the same words). `replay.py` has
  `run_replay`: every `(target, case)` pair (`base`, `tuned`, and `compare` with a compare
  target, whose identity is `compare.json`, frozen when attached and checked under the lock
  by `compare.attach_compare`; a run planned without an attached one is refused), each
  target on its own cases (`replays`, `Replays`: every case unless the caller gives fewer),
  generated,
  scored and judged on
  a thread pool, each generation appended to `generations.jsonl` and each result to
  `results.jsonl` as it completes, so a rerun skips finished pairs; `evaluation.json` is the
  identity a rerun must match (`evaluation_identity`; a mismatch is a `SettingsConflict`, exit 2,
  asking for a new `--out-dir`); a sampler or judge failure is an error result, counted apart and out of
  every rate, and a generation whose prompt the renderer refused (`RendererError`) is one a
  rerun never retries (`is_final`, counted as `errors_unrenderable`); `cancel` stops new
  pairs and keeps finished ones (any other error waits for the pairs in flight, then
  propagates). `summarize` writes
  `summary.json`: per-target pass rates (judge and metrics), per-trajectory agreement, the
  paired wins, regressions and ties, and `tuned_minus_base` with a seeded paired bootstrap
  95% CI over trajectories (`bootstrap.py`'s `paired_bootstrap_ci`), `tuned_minus_compare`
  the same way over the compare target's cases (both through `_paired`; each target counts
  the cases it does not replay as `skipped`), plus the agreement-not-task-success `CAVEAT`. A `Publisher` (default `NoopPublisher`) gets a complete evaluation and returns
  what it reports, the outcome's `publication`. `runner.py` is what `evaluate` and `train
  --evaluate` run: `plan_evaluation` (an `EvalPlan`: cases, judge, settings and `Publishing`,
  whether the dataset's push receipt allows publishing; cases the run's model cannot replay
  are left out, `excluded`, and the preview counts them as `excluded_cases` and
  `excluded_by_reason`, through `cases.py`'s `exclusions`), calibration (`prepare_eval_dir`
  stores a passing one), `replay` with the plan's publisher, `replay_result` and
  `evaluation_exit_code`, `plan_standalone` and `StandaloneEvaluation` for `evaluate`, and
  `EvaluateAfterTraining`, the `after_finalize` hook that evaluates in the training session
  and stops, partial, at the run's `--max-spend-usd`. `EvalPlan.evaluate_command` repeats
  the settings that differ from the defaults and the judge for `evaluate` (`standalone`), and
  pins every setting after `train --evaluate`, so its retry passes the identity check; a
  preview into an `--out-dir` that holds an evaluation with other settings fails (exit 2,
  with a free `--out-dir`) instead of offering a `--confirm` that would be refused. With
  `--compare-model` (or a `compare.json` the directory has), `EvalPlan.targets` adds
  `compare`, and `StandaloneEvaluation` opens a training session only when base or tuned has
  a case left to generate, and the compare model's client only when it has, next to the
  judge's before anything is written (`EvalPlan.work`, what the preview counts, says which
  have work left); `prepare_eval_dir` has it answer one `probe` under the directory's lock
  right before `compare.json` is written, so also after `check_identity` dropped the one
  attached. The preview names only the targets that generate (their provider, billing and
  recipients); the others' answers are reused. It counts the probe as one more compare call
  when the run will make it: no `compare.json`, or one `check_identity` drops
  (`EvalDir.replaces`, which `check_identity` itself decides by). `EvalPlan.replays`, the
  one place the skip rule is applied, are the cases each target replays (the compare model
  skips `EvalPlan.compare_skips`), passed to `run_replay` and to the publisher's
  `PublishSettings`; the preview's `compare` block counts the skipped (`skipped`,
  `skipped_by_reason`), the preview and the result warn about them, and a compare model that
  would be sent no case at all fails the preview (`compare.require_sent`, exit 2).
  `compare.py` is the compare target: `resolve_compare` (the judge spec grammar,
  `COMPARE_MAX_TOKENS` and the provider's lowest effort; the dev-only `fake-compare`; a stored
  spec is resolved as the flag is, never from the stored endpoint), `CompareTarget` (its
  identity, recipient and billing), `probe`, `stored_compare`, `compare_conflict` (the one
  check the preview and `attach_compare` share: what changed and a free `--out-dir`),
  `attach_compare` (writes `compare.json` under the lock), `ChatSampler` (a `Sampler` over any
  `ChatCompleter`, the prefix without `reasoning_content`), `FakeCompareSampler`, and
  `refused_prefix`, the pure check of whether a provider's chat API can carry a case's prefix
  (`REFUSED_SHAPES`: OpenAI and Anthropic refuse no messages, a tool call whose results do
  not open the turn right after it, and a tool result that answers no call of the turn
  before it; Anthropic also a prefix that opens or ends on an assistant turn; Anthropic is
  judged on the turns its client sends, `anthropic_messages.py`'s conversion; Fireworks and
  the fake take any), with `compare_skips` over cases; it takes the `compare.json` path and imports nothing from `replay.py`, which imports it.
  `benchmark.py` is `benchmark`: `plan_benchmark` (the test split's cases, with `--max-rows`
  only those of a seeded sample of its conversations, `sample_conversations`; the judge and
  one `Candidate` per base model, with its renderer identity and the cases of that sample it
  cannot replay, `excluded`; a model is named `PROVIDER/ALIAS` when another provider offers
  its alias, `Candidate.target`; a model whose renderer cannot load is planned as an error,
  `Candidate.unavailable`, its results not scored (planning reads the judge and evaluators it
  froze, and which of them have work left); a preview whose models share no case they all
  replay fails), `execute` (the judge calibrated once, then each model replayed in a
  session of its own, sized on its own cases, into `OUT/<model>` through `run_replay` with the
  model as its one target, so each resumes; each model's directory is identified by its own
  cases and renderer identity, `BenchmarkPlan.identity`, never by the other models, so adding
  or dropping a model reuses the finished ones; a model whose session fails is an error entry
  and the others still run; each model first copies what it takes from the dataset's store,
  so one with every answer reused opens no session, and shares its own answers after, a
  complete one included; `BenchmarkPlan.inputs` and `keys`, from each model's renderer
  identity, are read at plan time only when the store exists), `leaderboard` (a
  `Leaderboard`: each model's headline pass rate, the judge's unless `--evaluators` names
  another, with a bootstrap CI over trajectories on its own cases, its judge pass rate,
  `judged_cases`, `headline_cases` (the cases its headline scored, the rate's denominator) and
  `coverage` (their share of the test cases with a prompt), and its answers `truncated` at
  `--max-tokens` or `unparseable`, each a warning; the finished models, with no session error,
  nothing pending, no error a rerun retries and no score the headline still needs, ranked on
  the cases the headline scored for every one of them, `common_cases`, by their headline rate
  there, with each one's paired gap to the best; a model a rerun can still advance listed
  unranked, never narrowing that set; `rerun`, whether a rerun has work left, an evaluator
  score outside the headline included, but not a check that raised, whose frozen file raises
  again: the hint names it, as `evaluate`'s does) and `recommend` (`fine_tune`;
  `base_meets_target` when the best ranked model's own headline rate reaches
  `--target-pass-rate` and its `headline_cases` cover `TARGET_COVERAGE`, 90% of the test cases
  with a prompt; or `no_result`; the reason quotes that count and says when results are
  partial, and models within noise of the best are named).
  The package exports its names lazily, so importing the runner (the CLI does, for every
  command) loads no judge client. `targets.py` makes the two samplers from a session: the live training
  session's best sampler checkpoint, or the best training checkpoint loaded and saved again
  as an `ev...` sampler checkpoint in a new session (`run_samplers`), only for the targets
  with a case left to generate (`targets`). `reuse.py` is the dataset's store of base
  results, `DATASET/replay_cache` (`generations.jsonl` by `generation_key`: provider, base
  model, context, renderer identity (a family's `prompt_check` and `message_rewrite`
  included), `max_tokens`, prefix and tools; `verdicts.jsonl` by
  `verdict_key`: the judge's identity and exact request; append-only, first line of a key
  wins): `ReuseIndex.load` reads it once at plan time (`EvalPlan.reuse`, `None` with
  `--no-reuse`), `plan_target` is what a target takes (the preview's `reuse` block and
  `EvalWork.reused`), `import_target` copies it in after calibration passed
  (`prepare_eval_dir`; answers get `reused_from`, results `reused`), and `export_target`
  adds the run's own base answers and verdicts with a judge answer after the replay
  (`share_base`), under the store's lock tried once (a busy one is a warning naming the
  command that shares them), only from a directory whose `reuse.json` (`stamp`, written
  before its first answer) holds the current `key_inputs` (an evaluation made before stamps
  counts as version 1). Only base answers are shared; `evaluation.json` records
  `"reuse": false` only with `--no-reuse`, so a rerun without the flag is a conflict whose
  hint adds it first (`no_reuse_hint`, in `benchmark` too). `evaluators.py` is `--evaluators`
  (evaluate and benchmark): `load_spec` (the TOML spec through `tomllib` and pydantic,
  `extra="forbid"`), `JudgeEvaluator` (a rubric, `judge.py`'s `RUBRIC_PROMPT`, calibrated on
  the user's examples by `calibrate_examples`) and `CheckEvaluator`, each with the
  `identity()` frozen by name in
  `evaluators.json` (`attach_evaluators` under the lock: the list only grows, a change is a
  `SettingsConflict` naming a new version beside it or a free `--out-dir`), `resolve_evaluators`
  (the flag, which no `--confirm` command but the preview's repeats, since a run attaches the
  whole spec; else the stored record, in a benchmark the longest of its models' (the newest
  on a tie), which a complete model does not attach; a stored check loads only with cases
  left) and `scorers`, which `run_replay` runs on every pair
  missing an entry (`replay.outcome` reads a result for the judge, an evaluator or `all`).
  Evaluators are scored per directory, never shared through the store; their metrics go only
  with spans being published, and one added after is a warning.
- `providers/` - the provider seam. `base.py` has the neutral types (`ModelProfile`,
  `TrainSpec`, `CheckpointRef`, `StepMetrics`, ...) and the `Provider`, `TrainingBackend`,
  `TrainingSession` and `Sampler` protocols; `__init__.py` is the lazy registry
  (`register_provider`, `get_provider`, `known_providers`), which imports a provider only on
  first use; each entry names its `--provider-option` keys (`registered_option_keys`, which
  `train --help` lists; a test holds them to the option models) and its model table, each
  alias with every name that selects the profile (`ModelProfile.names`: alias, provider
  model id, tokenizer; `known_models`, `providers_for_model`, which `train` infers
  `--provider` from without importing a provider; a test holds it to the profiles, and
  every `resolve_model` goes through `base.find_profile`), and what it needs
  (`ProviderInfo`: its `label`, the `credential` it signs in with, the `packages` each step
  needs, the training flags it `refuses`, whether it `deploys`, whether it is `offline`,
  whether it trains in a job of its own (`remote`, for `train --remote`: `require_remote`);
  `tests/providers/test_provider_needs.py` holds them to the providers). The `train`,
  `resume`, `evaluate` and `deploy` previews call `require_packages`,
  `require_supported_flags` and `require_deploys` with it, so a missing SDK or a refused flag
  fails the preview (exit 2), never after `--confirm`; help lists `registered_providers`. `fake.py` is the offline, deterministic, dev-only reference provider (registered
  with `dev_only=True`, so it needs `PUP_FINETUNE_DEV=1`) that the default tests train
  against; its tuned checkpoints learn, per role of the message before each trained turn,
  whether to call a tool (by position) or answer in text, carried in the checkpoint URI. `options.py` has `parse_options`, which every provider validates its
  `--provider-option`s with (unknown keys refused, string values decoded as JSON). `fireworks/` is the Fireworks provider: its `__init__.py` only registers the
  `FIREWORKS_API_KEY` `CredentialSpec` (the registry imports it, so `auth` and `doctor` see
  the key without loading anything heavy; the validator lists accounts); `models.py` is the
  profile table (`qwen3p8-27b`, `glm-5p3-flash`, `kimi-k3`, `deepseek-v4-flash-0731`,
  `muse-glimmer-30b`; each `max_seq_len` the `trainingContextLength` Fireworks reports),
  each entry naming a rendering family that supplies its
  tokenizer and revision; `api.py` is a small read-only REST client over `httpx` (no SDK;
  `model_info`, `list_accounts`, injectable transport, redacted errors, 401/403 as
  `FireworksAuthError`, 404 as a missing model); `provider.py` has `FireworksProvider`
  (profiles, preflight against `GET /v1/accounts/fireworks/models/<id>`, the family's HF
  renderer, the `max_seq_len` option). `client.py` is the only module that imports the
  Fireworks Training SDK (`fireworks`, `tinker`; the `fireworks` extra), and only on first
  use: it defines the narrow interface the backend uses (`FireworksTraining` -> `service()` ->
  `TrainingService` -> `TrainingRun` and `TokenSampler`, with plain results such as
  `ForwardOutput` and typed `CheckpointRow`s), implements it as `FireworksTrainingClient`
  (training runs in the key's own account; the recorded account id is `FIREWORKS_ACCOUNT_ID`,
  else the key's first account), converts to
  `tinker.Datum` and `tinker.AdamParams`, and maps every SDK failure with `fireworks_error`
  (`ProviderCapacityError`, `ProviderAuthError`, `TransientError`, redacted). `backend.py`
  has `FireworksBackend` (`provider.backend()`, built lazily with a client factory, so the
  SDK loads only when a session opens): one serverless session per `open_session`, always
  ended on exit (in-flight results drained, or abandoned on an error), resume through
  `create_training_client_from_state_with_optimizer`; `FireworksSession` (the validation loss
  computed from `forward`'s per-token logprobs, checkpoint names checked against the
  17-character limit before saving, `promote` picking the newest promotable sampler row
  carrying the checkpoint's name (`promotable_row`) and adopting a model this run already
  promoted whose response was lost (`adopt_since`: the run's request to promote that
  checkpoint, saved in `state.json` under `promotions`, less a minute's clock allowance),
  sampler checkpoints of an ended session refused with a hint); and `FireworksSampler`
  (render, sample greedily to the renderer's stop tokens, parse; only sampling is retried, a
  bounded number of times). `client.py` also wraps
  the SDK's `Fireworks` client for deployments (`FireworksDeployments`,
  `FireworksDeploymentsClient`: `match_shapes`, `create_deployment`, `get_deployment` and
  `delete_deployment` with plain `ShapeVersion` and `DeploymentRecord` results, 404 as gone,
  409 as a taken id). `deploy.py` has `FireworksDeployer` (shape choice, create with hot
  loading off and 0 to 1 replicas, polling every 15 s until `READY`, `DeploymentNotReady`
  for a bad state or timeout, the smoke test through the inference API retried while it
  scales from zero, idempotent delete; injectable clock and sleep), which
  `FireworksProvider(deployer=...)` replaces in tests. `baseten/client.py` is the only module
  that imports the Baseten Loops SDK (`baseten-loops`; the `baseten` extra), and only on first
  use, after refusing the `LOOPS_*` environment overrides: `BasetenHTTP` (Baseten's REST
  routes, among them the checkpoint deployment, instance types, models by name and model
  deletion; `Bearer` auth, no redirects, a 4 MiB cap, fixed messages; 401/403 as
  `ProviderAuthError`, 429 as `ProviderCapacityError`, a 4xx to a create as
  `DeploymentRejected` with Baseten's reason, `TransientError` for a call safe to repeat, a
  create accepted with an unreadable answer or a deployment whose outcome is unknown),
  `loops_service` (a Loops session with the key, base URL and dedicated availability passed
  explicitly), `loops_trainer`, `loops_sampling_client`, `loops_failure` (SDK failures, with
  the same auth, capacity and transient classes) and `to_loops_datum`. `baseten/backend.py`
  has `BasetenBackend` (a key factory resolved when a session opens, the profile's renderer,
  an optional `max_seq_len` cap), `BasetenSession` and `BasetenSampler`: one dedicated Loops
  trainer per session (its context the next power of two over the longest example, at least
  1024; no capacity is a `ProviderCapacityError` with nothing left running), steps as SDK
  futures, `bt://loops:` checkpoints, resume with the optimizer state, no promotion, and
  samplers on their own deployments with the profile's context (render, sample greedily,
  parse with the tools; a completion cut at `max_tokens` is not `format_valid`; only sampling
  is retried, a bounded number of times; `generate` is safe from several threads); leaving a
  session always deactivates its run and samplers. `baseten/__init__.py` only registers the
  `BASETEN_API_KEY` `CredentialSpec`, as Fireworks' does (falling back, read only, to the key
  of the default remote in `~/.trussrc`; the validator lists models). `baseten/provider.py`
  has `BasetenProvider`: the `qwen3p5-9b`, `glm-5p3-flash`, `qwen3p8-27b` and `kimi-k3`
  profiles, each
  with its own context (tokenizer pins from
  their families), a read-only `preflight` (the SDK loads, and Loops' capabilities offer and
  enable the model with a limit of at least the `max_seq_len` option, else the profile's
  context), the family's HF renderer and the `max_seq_len` option (at least 1024; it
  cuts the training profile, so it also sizes the samplers and an evaluation's trainer);.
  `baseten/deploy.py` has what its `deploy`, `smoke_test`, `owns_deployment` and `undeploy`
  run: lookups first (the checkpoint record, synced and of a served model; the instance type
  with exactly the shape's GPUs; no model already named the deployment id), then
  `POST /v1/loops/checkpoints/deploy` sent once, as a model named the deployment id (a 4xx is
  `DeploymentRejected`; after any other failure the model is read back by name and adopted
  only when its one deployment serves the checkpoint on the instance type, else refused as
  taken, else `TransientError`), polling every 15 s until `ACTIVE` (`DeploymentNotReady` for
  a failed status or the timeout), the replicas set to 0 to 1, a smoke test of one chat
  completion naming the checkpoint on the deployment's `/sync/v1` route, retried while it
  scales up, and an idempotent `undeploy` that deletes the model. `deployment_name` is
  `None` (Baseten assigns the ids), and the `Endpoint`'s `name` is
  `models/<model id>/deployments/<deployment id>`. `baseten/jobs.py` is `train --remote`'s
  `BasetenJobs` (`provider.remote_jobs()`): the plan (the CLI and `uv`, the team, its
  `pup_finetune_api_key` secret, no project of the run's name with jobs), one `baseten loops
  exec` with the key in its environment only (exit codes and HTTP statuses mapped to what was
  or may have been created), and the reads of the job, its logs page by page and its trainer.
  See `docs/architecture.md`.

Tests live under `tests/` and mirror `src/`. Shared fakes live in `tests/fakes/` (import them
as `tests.fakes...`); `tests/fakes/pup.py` has `FakePup`, a fake pup runner that answers by
argument prefix (or computes the answer with a `handler`) and records every call;
`tests/fakes/llmobs.py` has `FakeExportApi`, the Export API search route served by cursor.
`tests/fakes/datasets.py` has `FakeDatasetsApi`, in-memory projects, datasets and records
behind the pup calls `LLMObsDatasets` makes, with failures to inject.
`tests/fakes/pushed.py` has `mark_pushed`, which writes the push receipt a complete push of
every kept row would leave, so a real-provider `train` test runs on an imported dataset.
`tests/fakes/text.py` has `letters`, a number spelled in letters, for row texts that must not
be copies of each other (request grouping folds numbers to `0`).
`tests/fakes/experiments.py` has `FakeExperimentsApi`, in-memory experiments and their events
behind the pup calls `LLMObsExperiments` makes (failed and lost creates and event posts,
delayed or capped indexing); it chains to the `api` handler installed before it, so it serves
one `FakePup` with `FakeDatasetsApi`.
`tests/fakes/tokenizer.py` has `FakeChatTokenizer`, a character-level chat tokenizer with a
Qwen3.8-like Python template (optionally merging `\n\n` across the prompt boundary) or a
GLM-5-like one (`glm5_like`), and extra special tokens (`DEEPSEEK_SPECIALS`) for a fake behind
an encoder; `FakeXtmlTokenizer` renders Kimi K3's XTML as its tokenizer code does, control
tokens apart from message text (`render_tokens`).
`tests/fakes/rows.py` has `import_with_edited_test_row`, the tool-use fixture rows imported
with one test-split row edited, and two edits: `import_with_refused_test_row` (a prompt the
fake renderer refuses) and `import_with_opening_assistant_test_row` (a row that opens with
an assistant turn).
`tests/fakes/chat_api.py` has `FakeChatApi`, an OpenAI-compatible chat completions endpoint
(scripted responses or errors first, then an answer function) and `matching_judge`.
`tests/fakes/fireworks_api.py` has `FakeFireworksApi`, the Fireworks model and account routes
behind an `httpx.MockTransport`, with statuses and transport errors to inject.
`tests/fakes/fireworks.py` has `FakeFireworksTrainingClient`, the Fireworks Training API
behind the `client.py` interface in memory (sessions, server-renamed checkpoints, promotion
that fails once a session has expired, deterministic losses, canned samples, injectable
capacity, auth, 5xx and connection failures); `FireworksProvider(training_client=...)` trains
against it (see `tests/providers/fireworks/test_fireworks_backend.py` and
`tests/cli/test_train_fireworks.py`).
`tests/fakes/fireworks_deploy.py` has `FakeSdkFireworks`, the SDK's `Fireworks` client as far as
deployments use it (shape matching, deployments that report scripted states, injectable
failures), `FakeClock`, and `fake_deployer`, the provider's `deployer` factory on them and a
`FakeChatApi` (see `tests/providers/fireworks/test_fireworks_deploy.py` and
`tests/cli/test_deploy.py`).
`tests/fakes/baseten.py` has `FakeLoops`, the Loops SDK in memory (sessions, runs, futures,
deterministic losses and logprobs, `bt://loops:` checkpoints with the name limit, expiry,
closed clients and deactivated runs, samplers that answer a fixed completion, injectable
failures), and `BasetenTransport`, Baseten's HTTP routes by `"METHOD /path"`
(`FakeLoops.transport()` also serves its runs, samplers and their deployments, checkpoint
records, instance types, checkpoint deployments as models with scripted statuses,
autoscaling and config, model deletion and each deployment's chat route, with `fail`, `lost`
and `after_deploy` to inject failures before or after a create took effect);
`capabilities()` builds a Loops capabilities answer; `FakeTrainingJobs` has the teams, team
secrets, training projects, jobs and job logs `train --remote` reads, whose `routes()` merge
into `FakeLoops.transport()`. `tests/fakes/baseten_cli.py` has `FakeBasetenCli`, the `baseten`
CLI as a `CliRunner` that records each call with its upload tree and answers as scripted.
`tests/fakes/keyring.py` has in-memory, failing and locked keyring backends; every test starts
on the failing one, so none touches the real keychain, and with every registered provider
key variable cleared (`live` tests keep them), so no key in the developer's shell passes a
test that fails in CI. The conftest's `judge_keys` fixture
puts a key for every judge provider in the environment, for tests that preview a paid judge
or compare model whose client they fake.
`tests/providers/contract/` has the `RendererContract`, `BackendContract` and
`ProviderContract` suites every provider subclasses (`tests/providers/test_fake_contract.py`
runs them against the fake provider; `tests/providers/baseten/` against Baseten on its fake;
`tests/providers/fireworks/` against Fireworks on its fakes; `tests/rendering/test_hf_contract.py`
runs the `RendererContract` on each family's real tokenizer under `-m tokenizer`).
Synthetic fixtures live in `tests/fixtures/` (see its README); `rows/` holds canonical rows and
`llmobs/` raw Export API pages.

## Conventions

- Every first-party file starts with the Apache-2.0 license header (see any `.py` file); shell
  scripts and YAML use the same lines with `#`.
- Everything is typed; `mypy --strict` must pass on `src/`.
- Each command writes exactly one JSON result to stdout; progress and logs go to stderr, and
  after the result stderr gets every warning in it and the command's closing summary (a
  preview prints its card first, generated from the preview by `_summary.preview_card`, at
  most seven lines, dropping lower-value rows rather than the cost, billing, recipients or a
  refusal, then its warnings, and its `Next` line last). The
  exceptions print text on stdout: `help`, `--help`, `--version`, a group's help when run
  bare (exit 2) and `dataset triage --print-template`; `docs/commands.md` lists them. The
  hidden `_worker --report-state` prints only its state report lines on stdout. Hints
  that send the reader to a command's help name `<prog> help <command>` (`prog_name`:
  `pup finetune` under pup, `pup-finetune` standalone), never `--help`. Wrap
  commands with `run_command` from `cli/_common.py`: return the result mapping and raise a
  `FinetuneError` subclass for expected failures. Return `CommandResult(data, exit_code=1)`
  when the result must still be printed but the command should fail (e.g. `doctor`).
- Exit codes: `0` success; `2` when the user can fix it by changing the command or the
  directory (usage, validation, a missing `--confirm`, read-only, a wrong, missing or
  non-empty directory, a frozen-settings conflict, the wrong phase); `1` when something
  outside failed or local state is broken (a provider or pup failure, corruption, a lock held
  by a live process, an interrupt). Raise the `errors.py` class whose `error_code` names the
  mistake; add a code to `ERROR_CODES` only when no existing one fits (the codes are a public
  contract). `tests/cli/test_exit_codes.py` runs common mistakes through the CLI and checks
  each exit code and error code.
- Paid or remote-changing actions preview by default and run only with `--confirm`. `--confirm`
  is never implied by `PUP_AUTO_APPROVE` or agent mode.
- A result's `next_command` is a command that runs exactly as written (after the user's yes,
  when it ends in `--confirm`), or `null`: never a placeholder, `...`, a file that does not
  exist, a deletion of what the command just made, or a `--confirm` the preview's own checks
  know will be refused (fail the preview with that error instead, exit 2). Choose it from the
  current state. Alternatives, and commands that need a value only the user can supply, go in
  `next_steps`, a list of `{"command", "why"}` whose commands mark such values in one style,
  `<UPPER_CASE>`; never copy a step into `next_command`. A preview's `next_command` repeats
  the flags that were typed plus the identity it must pin (`--run-id` for `train`, the judge
  for `evaluate`), not every resolved default. Build them with `next_steps.py`; the test
  conftest checks every emitted result against the rule.
- A preview is the approval: `action` (one sentence of what `--confirm` does), `billing`
  (the provider's sentence: what it bills for and the usage counted, never a dollar figure
  we cannot compute), `data_recipients`, `rows`, `settings` (the resolved values that will
  be used, each value a default resolved to with its source, e.g. `rank_source`),
  `defaults_used` and `warnings`.
- Honor `PUP_READ_ONLY`: reject every remote write and paid run when it is set. Previews still
  run (they make no remote write) and say so: `"read_only"` in the preview and `next_command`
  null. A run that never leaves the machine (the offline fake provider and judge) is allowed.
- Never ignore what the user asked: an option that does nothing without another one is a
  usage error (add it to its command's table for `_flags.require_enablers`), and a preview
  names every recipient of the data it would send in `data_recipients`.
- Only `datadog/pup.py` spawns `pup`. Only `providers/baseten/jobs.py` spawns `baseten`.
  Only `providers/<provider>/client.py` imports a training
  provider's SDK, and only `anthropic_client.py` imports `anthropic` (and `httpx2`).
  Only `rendering/hf.py` imports `transformers`, and only when a tokenizer is first loaded.
- Secrets are never printed, logged or written to disk, and never accepted as a flag value.
- Tests use fakes, not mocks. Default tests make no network or paid calls.
- Fixtures are synthetic only: never commit real trace content or customer data.
- No internal links or internal-only references; this repository will be public.

## Adding a provider

See "How a new provider plugs in" in `docs/architecture.md`.
