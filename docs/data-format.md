# Data format

This page describes the training row every provider trains on, how an LLM span from Datadog
Agent Observability becomes one, why a row can be left out, what a dataset directory holds,
and the record `dataset push` sends to an Agent Observability dataset.

## The canonical row

A dataset holds one row per conversation, in OpenAI chat format plus where it came from. The
schema lives in [`src/pup_finetune/data/schema.py`](../src/pup_finetune/data/schema.py), and its
version is `1`. `dataset import` accepts rows in this shape, one JSON object per line.

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | A unique, non-empty row id. |
| `messages` | list | The conversation, with roles `system`, `user`, `assistant` and `tool`. |
| `tools` | list | The tools the conversation declares, as `{"type": "function", "function": {"name", "description", "parameters"}}`. Optional. |
| `source` | object | Where the row came from. Every field is optional. |
| `schema_version` | integer | `1`. Optional. |

The messages follow these rules:

- A `system` message may only come first, and the last message is from the `assistant`.
- An assistant message may carry `tool_calls`, each `{"id", "type": "function", "function":
  {"name", "arguments"}}`, where `arguments` is a JSON object encoded as a string.
- An assistant message may carry `reasoning_content` and `weight`: `0` means "do not train on
  this turn", `1` or no weight means train on it.
- A `tool` message answers an earlier call through `tool_call_id`, once.
- When the row declares tools, every called tool must be one of them.
- A `null` content is read as `""`, and an empty `tool_calls` list as none. Nothing else is
  repaired: a row that breaks a rule is excluded with a code.

`source` holds `trace_id`, `span_id`, `ml_app`, `model_name`, `start_ns` (the span's start in
nanoseconds), `evaluations` (the span's evaluations by name) and `tags`.

A synthetic example, formatted over several lines here (a file holds one row per line):

```json
{
  "id": "row-weather-001",
  "messages": [
    {"role": "system", "content": "You are a weather assistant. Use the tools for live data; answer briefly."},
    {"role": "user", "content": "What's the weather in Northvale right now?"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_w001a",
          "type": "function",
          "function": {"name": "get_weather", "arguments": "{\"city\": \"Northvale\", \"unit\": \"celsius\"}"}
        }
      ]
    },
    {"role": "tool", "tool_call_id": "call_w001a", "content": "{\"city\": \"Northvale\", \"temp_c\": 14, \"conditions\": \"light rain\"}"},
    {"role": "assistant", "content": "It's 14°C with light rain in Northvale."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Current conditions for a city.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
          "required": ["city"]
        }
      }
    }
  ],
  "source": {
    "trace_id": "synthetic-trace-0001",
    "span_id": "synthetic-span-0001",
    "ml_app": "synthetic-weather-bot",
    "model_name": "example-model-small",
    "start_ns": 1767225601000000000,
    "evaluations": {"helpfulness": "good"},
    "tags": ["env:synthetic", "ml_app:synthetic-weather-bot"]
  },
  "schema_version": 1
}
```

## From an LLM span to a row

`dataset pull` reads LLM spans through the Agent Observability Export
API. Each span becomes one conversation: its input messages followed by its output messages
([`src/pup_finetune/data/convert.py`](../src/pup_finetune/data/convert.py)).

| LLM span field | OpenAI chat field |
| --- | --- |
| message `{role, content}` | The same role and content. A missing role is `user` on input and `assistant` on output. |
| message `role: "reasoning"` | Dropped by default. With `--reasoning preserve`, the `reasoning_content` of the next assistant message in the same list (or the previous one when none follows). |
| `tool_calls[{name, arguments, tool_id}]` | `tool_calls[{id: tool_id, type: "function", function: {name, arguments}}]`, with `arguments` as a compact JSON object string. |
| `tool_results[{result, tool_id}]` | One `{role: "tool", tool_call_id: tool_id, content: result}` per result, before the message's own content. |
| `tool_definitions[{name, description, schema}]` | `tools[{type: "function", function: {name, description, parameters: schema}}]`. |
| trace id, span id, `ml_app`, model, start, evaluations, tags | `source`. |

A content list is joined when every part is text. An assistant message with neither content
nor tool calls is dropped. Within a trace, a conversation that is a prefix or a duplicate of
another is dropped, so each sub-agent keeps its own row. A row's split comes from its trace,
so every row of a trace lands in the same split, and copies of one request share it too: rows
whose user messages are the same once lowercased, with numbers as `0` and whitespace collapsed,
join their traces into one group, and a group gets one split. A request asked in more traces
than 5% of them (rounded up) and more than 5 is generic and joins nothing: a fixed prompt many
conversations share, such as "Summarize the above", keeps a split per trace, with a warning
naming some of its rows. Groups have no size cap, so in a dataset of 50 traces or more, a
group of 3 traces or more holding more than 10% of the rows is warned about with a few of its
rows: its traces are likely chained through requests they share. A trace keeps its split once assigned; a new trace joining copies already in two splits
takes the split holding most of their kept rows. Whenever the kept rows of a group are in more
than one split (copies recorded apart before they were grouped), the summary's
`split_conflicts` counts the groups and the rows outside their group's split, names up to 10 of
them per split, and names `split_conflicts.jsonl`, a drop label for each of them that `dataset
label --file` takes (the result's first next step). `split_groups` in a dataset's summary is
how many groups its traces form. In a dataset of fewer than 50 traces, a validation or test
split more than twice its fraction, or 10 or more points off it, is warned about (`validation
has 5 of 14 rows (36%, 4 of 13 traces) though --validation-fraction is 0.1 ...`): splits are
assigned per trace and request, so small datasets vary, and pulling more rows evens them out
(the seed of a dataset is frozen).

`--from` and `--to` take ISO 8601, epoch milliseconds (a bare number is milliseconds, while
pup reads one as seconds; one below 10^11, such as `1728000000` or `20261001`, is refused
with a hint at what it may mean), or a relative time: a number and a unit (`7d`, `24h`, `30m`, or
pup's spelled-out `5min`, `2hours`, `5 minutes`), `now-<n><unit>` or `now`. A relative time
is frozen as `now` or `now-<n><unit>` (`7d` is `now-7d`), with the window it resolved to.

## Why a row is left out

A row is excluded as a whole, never repaired, and the reason goes to `excluded.jsonl` with a
stable code. Schema codes come from the row rules above; dataset codes are reasons the schema
cannot see. `dataset show` counts them.

<!-- BEGIN GENERATED: exclusion codes (scripts/gen_command_docs.py) -->
| Code | Kind | Why the row was left out |
| --- | --- | --- |
| `no_assistant_message` | schema | The conversation has no assistant reply to train on (for example a span with no output). |
| `system_not_first` | schema | A system message appears after the start of the conversation. |
| `tool_result_without_call` | schema | A tool result does not answer any tool call made earlier in the conversation. |
| `duplicate_tool_call_id` | schema | Two tool calls in the conversation share an id. |
| `tool_not_declared` | schema | The model called a tool that is not in the span's tool definitions. |
| `last_message_not_assistant` | schema | The conversation does not end with an assistant message. |
| `invalid_arguments_json` | schema | A tool call's arguments are not a JSON object. |
| `invalid_row` | schema | The row does not fit the canonical schema. |
| `duplicate_conversation` | dataset | The same conversation is already in the dataset under another id. |
| `non_text_content` | dataset | A message holds non-text content, such as an image; only text conversations are used for training. |
| `unstructured_tool_message` | dataset | The app writes tool calls as text; these conversations can't be rendered as tool turns. |
| `span_error` | dataset | The LLM call ended in an error, so it is not a completed model turn; pass --include-errors to keep errored calls. |
| `contained_conversation` | dataset | A later pull read a longer conversation of the same trace that contains this one; the longer row trains on the same turns. |
| `unreadable_span` | dataset | The exported span does not have the fields of an LLM span (such as its trace id), so it could not be converted. |
| `unreadable_record` | dataset | The dataset record's input is not chat messages, or its expected output is not an assistant message, so it is not a conversation to train on. |
| `split_conflict` | dataset | The dataset record names another split than a record of the same trace does, and a trace is never in two splits. |
| `trace_row_cap` | dataset | Its trace already gives --max-rows-per-trace rows, so one trace cannot dominate the dataset. |
<!-- END GENERATED: exclusion codes -->

`train` can also leave rows out when it renders them for the model. These rows stay in the
dataset; the training preview lists them with these codes:

| Code | Why the row is not trained on |
| --- | --- |
| `exceeds_context` | A rendered example of the row has more tokens than the model's training context. |
| `no_trainable_tokens` | Every assistant turn has `weight` `0`, so the row would train on nothing. |
| `template_error` | The model's chat template failed on the conversation. |
| `prompt_mismatch` | A turn's render does not continue its own generation prompt. |
| `unsupported_message` | The conversation holds something the chat template cannot express. |

## The dataset directory

A dataset directory is created `0700`, and its files `0600`, because they hold trace
content. Keep it out of version control.

| File | What it holds |
| --- | --- |
| `settings.json` | The source (`import`, `datadog`, or `datadog_dataset` for `dataset fetch`, which freezes the dataset's project, name, id and version) and its settings, such as the span filter and time window, plus the split settings. Frozen on first use: later runs may leave a flag out (it takes the frozen value), but may not pass another value (later imports pass the same split settings). |
| `rows.jsonl` | The accepted rows, one canonical row per line. Never rewritten by labeling. |
| `excluded.jsonl` | One `{"code", "detail", "row_id", "trace_id", "span_id", "line_sha256"}` per excluded row (fields that do not apply are left out). |
| `labels.jsonl` | Keep/drop decisions appended by `dataset label`, `dataset triage` and `dataset pull`: `{"id", "keep", "reason", "by", "at"}`, plus `"carried_from"` on a drop a pull carried, where `by` is `manual` (recorded with `dataset label`), `judge` (decided by the `dataset triage` council) or `check` (a row one of `dataset triage --checks` failed, with the reason `check <name>: <reason>`); triage never relabels a manual row. Older files that say `agent` or `human` read as `manual`. The last label of an id wins, a reason is required to drop, and a dropped row never trains and is never pushed. When a later pull retires a dropped row into a row that holds its conversation (a longer one of its trace, or one it now repeats, followed to the row that survives when the pull retires that one too), the pull appends a drop for that row with `"carried_from": "<id>"`, the reason `carried from row '<id>', which this row absorbed: <reason>` and `by` `manual` whoever dropped that row, so triage never relabels it (and `--agreement` does not compare with it, as the decision was made on the other row), unless that row is dropped already or labeled `manual`; a keep does not carry. `dataset label --file` refuses a line naming `carried_from`. |
| `splits.json` | The split of every trace (one per group of traces and copies of one request) and the frozen seed and fractions. A trace never changes split once assigned. |
| `summary.json` | Counts, splits and warnings, rewritten after every change. |
| `split_conflicts.jsonl` | While a group of copies has kept rows in more than one split, a drop label (`{"id", "keep": false, "reason"}`) for each kept row outside the split holding most of the group, for `dataset label --file`; rewritten with `summary.json`, and removed once there are none. |
| `raw/` | For a Datadog pull, the Export API pages it read, so a rerun resumes without fetching them again. |
| `pull_limits.json` | For a Datadog pull, the last pull's `--target-count` and `--max-candidates`, which a rerun without them continues toward. Not frozen. |
| `judgments.jsonl`, `triage.json` | For `dataset triage`, every judge vote (`{"row_id", "row_sha256", "judge", "criteria_sha256", "keep", "reason", "at"}`, plus `"too_long": true` when the judge refused the row as longer than its context window, which drops it) and the frozen rubric hash, judges and the settings they are called with, plus `checks_file`, the absolute path of the last council's `--checks` file (not frozen), which the step to finish the triage passes again. |
| `push.json` | The receipt `dataset push` writes (see below). |
| `replay_cache/` | Base model answers and judge verdicts shared by the evaluations and benchmarks of the dataset: `generations.jsonl` (`{"key", "message", "format_valid", "prompt_tokens", "completion_tokens", "latency_s", "stop_reason", "sampled_at", "source"}`, `source` the `{"command", "run_id"}` that made it) and `verdicts.jsonl` (`{"key", "judgment", "judged_at"}`). The keys are hashes of what the model or the judge is sent and how; append-only, the first line of a key wins. Safe to delete: it only costs new answers. A run that reuses a line copies it into its own directory with `reused_from` (an answer, keeping its `sampled_at`) or `"reused": true` (a result, its metrics computed again). |

**Editing a dataset.** Only `dataset import`, `pull`, `fetch`, `label`, `triage` and `push`
write these files. Each command checks what it reads: every row valid and under an id of its
own, and every kept row with a split; `dataset show`, `push`, `train`, `benchmark` and
`evaluate` also refuse two kept rows that hold one conversation (`Error [state]`, with the
fix). A label of an id that names no row changes nothing, and the summary warns about it. An
edit to a pushed row makes it `changed` (`pushed_with_changed_rows`), so `train` and
`benchmark` on a real provider need the dataset fetched first. Never edit `splits.json`: it is
the only record of each split (groups of copies and fetched records take splits the hash
would not give). To prepare rows your own way, analyze a copy of `rows.jsonl`, write the rows
you change, with new ids, to a file and `dataset import` it into a new directory; check that
`import.excluded` is 0, as a pandas round trip can write `null` tools or source. Never copy a
directory that has a `push.json`: the copy pushes into the same dataset.

## The pushed record

`dataset push` adds one record per kept row to an Agent Observability dataset
([`src/pup_finetune/data/push.py`](../src/pup_finetune/data/push.py)). The record holds what
the final assistant turn answers:

```json
{
  "input": {
    "messages": ["... every message before the final assistant message ..."],
    "tools": ["... the row's tools ..."]
  },
  "expected_output": {"role": "assistant", "content": "It's 14°C with light rain in Northvale."},
  "metadata": {
    "row_id": "row-weather-001",
    "split": "train",
    "trace_id": "synthetic-trace-0001",
    "span_id": "synthetic-span-0001",
    "ml_app": "synthetic-weather-bot",
    "source_model": "example-model-small",
    "schema_version": 1
  },
  "tags": ["split:train", "source:pup-finetune"]
}
```

The receipt, `push.json`, records the project (`project`, `project_id`), the dataset
(`dataset_name`, `dataset_id`, `dataset_version`), the record id of every row pushed
(`record_ids`, by row id), the batch being sent (`in_flight`, `in_flight_at`), when records
were last added (`pushed_at`), a random `push_id` written into the dataset's description, and
the hash of the dataset settings. It is saved after every batch, so pushing again sends only
rows not yet pushed and an interrupted push resumes without adding a row twice, and it names
the dataset, so a later push needs no `--name` or `--project`. `evaluate` publishes its Agent
Observability experiments on the dataset and version this receipt names.

## The fetched dataset

`dataset fetch` reads an Agent Observability dataset's current records back into a dataset
directory ([`src/pup_finetune/data/fetch.py`](../src/pup_finetune/data/fetch.py)), so a
dataset curated in Datadog is what trains. Each record becomes the row the pushed record
describes: its `input.messages` followed by its `expected_output` (which must be an
assistant message), its `input.tools`, the id `metadata.row_id` (else the record id, which
a copy of a record with a lower id also gets) and the source in `metadata`. The split is `metadata.split` or a `split:<name>` tag; a record without
one is assigned by its trace. A record that is not a conversation is excluded as
`unreadable_record`, and one naming another split than a record of the same trace as
`split_conflict`. The fetch also writes the dataset's `push.json`, with every row's record,
so `evaluate` publishes onto that dataset and `dataset push` has nothing to send.

## The evaluation directory

An evaluation directory (`RUN_DIR/eval` by default) holds `evaluation.json` (the identity a
rerun must match), `generations.jsonl` and `results.jsonl` (one line per target and case,
appended; the last line of a pair wins), `calibration.json` and `summary.json`, and, once
published, `experiments.json` and `published.json`. `reuse.json`, written before the first
answer, records what its base answers are sampled under (every input of their
`replay_cache` key but the case); they are shared to the dataset's `replay_cache` only while
it matches, so a directory resumed after an upgrade that changes them shares none. An
evaluation directory made before `reuse.json` shares as one made under the first release
that writes it (its `evaluation.json` pins every other input); a benchmark model directory
made before it shares nothing.

With `evaluate --compare-model`, `compare.json` records the compare target, written once it
has answered one short probe and never changed: `{"spec", "model": {"provider", "model",
"base_url", "credential", "temperature", "max_tokens", "reasoning_effort"}}`, where
`max_tokens` is 16000. A rerun resolves the spec again and refuses a directory whose
`compare.json` no longer matches. Its pairs go into the same files with `"target": "compare"`;
`evaluation.json` is unchanged.

`summary.json` has, per target under `targets`, the judge pass rate, every metric, errors by
stage, `errors_too_long` (requests longer than a context window, no retry fixes them) and
`errors_too_long_by_stage` (the same requests by the stage that sent them: `generation` for
the target's own model, `judge` for the judge),
`truncated` (generations cut off at their limit), `refused` (answers the provider refused)
and `reused` (`{"generations", "verdicts"}` copied from the dataset's `replay_cache`).
With base and tuned, `paired` and `tuned_minus_base`; with a compare target,
`tuned_minus_compare` (`judge_pass_rate`, `ci95`, `n_cases`, `n_trajectories`, `resamples`,
`seed`, and its own `paired` counts), over the cases the compare model was sent. Each target
also has `skipped`, the cases it did not replay, in none of its rates: `0` for base and
tuned, and for the compare model the cases whose conversation its API refuses every time.
The `evaluate` preview counts those by reason under `compare.skipped_by_reason`:
`no_messages` (nothing to send before the action), `starts_on_assistant` (on Anthropic, a
conversation whose first turn is the assistant's), `unanswered_tool_call` (a tool call with
no tool result right after it), `orphan_tool_result` (a tool result with no call right
before it) and `ends_on_assistant` (on Anthropic, a conversation that ends on an assistant
turn).

## Evaluator specs and checks

`evaluate --evaluators FILE` (and `benchmark --evaluators FILE`) reads a TOML spec, its paths
relative to it: an optional `headline` (`judge`, the default, an evaluator's name, or `all`)
and one `[[evaluator]]` table per evaluator. Every evaluator has a `name` (lowercase letters,
digits and `_`, starting with a letter, at most 40 characters; not `judge`, `all`,
`judge_pass`, `trajectory_agreement`, `completed_actions` or a metric's name) and a `type`. A
`judge` has a `rubric`, an optional `model` (named as `--judge-model`; the evaluation's judge
by default) and optional `examples`, a JSON Lines file of graded answers
(`{"conversation", "candidate", "pass"}`, with optional `tools` and `reference`; `candidate`
a string or an assistant message), at least one with `"pass": true` and one with
`"pass": false`. A `check` has a `file` and an optional `function` (its name by default):
`function(case, candidate)` gets copies of the case (`id`, `row_id`, `trace_id`,
`message_index`, `prefix`, `tools`, `reference`, `future_tool_results`) and of the generated
assistant message, and returns `True`, `False`, `None` (does not apply) or `(bool, reason)`.
An unknown key, a repeated or reserved name, or a headline that names no evaluator is a usage
error.

The evaluation directory then holds `evaluators.json`: `{"evaluators": [...], "headline",
"spec"}`, each evaluator `{"name", "type": "judge", "rubric_sha256", "rubric", "model"}`
(`model` as in `compare.json`) or `{"name", "type": "check", "function", "file_sha256",
"file"}`. Evaluators are added, never changed or dropped: a changed rubric, model or check
file (its whole sha256) is a `settings_conflict`, while the same ones attached from a spec
elsewhere take its `file` paths. Each line of `results.jsonl` gains
`evaluators`, by name: `{"status": "ok" | "not_applicable" | "error", "passed", "reason",
"raw", "no_verdict", "error"}`. `calibration.json` gains `evaluators`, each judge's report on
its examples. `summary.json` gains, per target, `evaluators.<name>` and `evaluators.all`
(`{"pass": {"rate", "passed", "n"}, "not_applicable", "no_verdict", "errors", "pending"}`),
and at the top `evaluators` (`{"headline", "names", "tuned_minus_base", "tuned_minus_compare"}`,
each by name: `{"delta", "ci95", "n_cases", "n_trajectories", "wins", "regressions", "ties",
"both_error", "one_error", "pending", "not_applicable"}`). `all` passes a case when the
judge passes it and every evaluator passes or does not apply; a case with an evaluator error
is left out of it. `published.json` keeps, after a span's digest, `+` and the evaluators it
was sent with.

## Row checks

`dataset triage --checks FILE` runs every public function that FILE defines (not one it
imports), in definition order, on each row it reviews, before any judge call. A function
takes one argument, a copy of the row as `rows.jsonl` stores it (see
[The canonical row](#the-canonical-row)), and returns `True`, `False`, `None` (does not
apply) or `(bool, reason)`. Anything else it returns, and anything it raises, is a usage
error naming the check and the row. Checks run in this process, with your privileges, one at
a time, and cost nothing.

A row that a check fails is labeled drop `by: check`, with the reason `check <name>:
<reason>` of the first check it fails, and is never sent to a judge. With `--checks` alone,
no judge is called, so the rows are labeled at once, without `--confirm`. A triage without
`--checks` skips the rows a check labeled. With `--agreement`, the checks are compared with
your labels apart from the council.

The file is not frozen. `triage.json` records its absolute path as `checks_file`, and an
edited file applies the next time a triage passes it. A label is never withdrawn, so a row
that an earlier version of a check dropped keeps that label after every check passes it. The
result counts those rows in `stale_check_labels`. Relabel them with `dataset label`, or add
`--rubric` so the council judges them.

## Where a dataset stands

Every dataset command's result (`import`, `pull`, `show`, `sample`, `label`, `triage` and
`push`) holds a `pipeline` block, read from the files above, and its `next_command` and
`next_steps` follow from it:

```json
{
  "rows": {"accepted": 14, "kept": 13, "dropped": 1, "unlabeled": 12},
  "triage": "incomplete",
  "push": {"status": "pushed_with_new_rows", "pending": 3, "changed": 0}
}
```

`triage` is `none` before any council triage (check labels alone leave it there), `complete`
once every row has a label (a check's included) and every judge label is backed by the votes
under the frozen rubric and judges, `unreadable` when
`triage.json` or `judgments.jsonl` cannot be read, and `incomplete` otherwise (for example
after `--max-rows`). `push.status` is `not_pushed` without `push.json`, `pushed` when it lists
every kept row as it is now, `pushed_with_new_rows` when `pending` kept rows are not pushed yet
(or a push did not finish), `pushed_with_changed_rows` when `changed` kept rows differ from the
records pushed for them (a push only adds records, so the dataset holds them as they were
pushed: the gate names up to 5 of them, and its step fetches that dataset into a new directory
to train on what Datadog holds; the fetch also brings back the records of rows dropped or
retired since the push, which the step counts and names up to 5 of, to drop again with
`dataset label`), and
`unreadable` (with `pending` and `changed` null) when `push.json` cannot be read. `train` and
`benchmark` on a real provider read only a `pushed` dataset. An `unreadable` part comes with a
warning in the result that names the file and the problem.

## What a command prints, and what its files keep

Each command prints one JSON result on stdout, and it differs from the files the command
writes. The result holds what decides the next step: keys in the order they are read,
measurements (losses, rates and their deltas and intervals, elapsed times, ETAs, spend
estimates, ages) rounded to 4 significant digits while values you set (flags, `settings`)
read exactly as given, and paths relative to the current directory when they are
inside it (else absolute). The files keep everything, at full precision and as recorded
(`run.json` names the dataset by its absolute path). A result's `files` names where the rest
is:

| Command | It prints | The rest is in |
| --- | --- | --- |
| `train`, `resume` | How the run ended (`status`), the best epoch and its validation loss, each epoch's losses (`best: true` on the best one), the promoted model, the deployable checkpoint's name, the elapsed seconds and what next. | `files.result` (`result.json`: every checkpoint with its URI, and each epoch's checkpoints and tokens) and `files.epochs` (`epochs.json`). |
| `status` | The phase, epoch and step against their maximums, the last training loss, the validation losses, the best epoch, the elapsed time and ETA, the promoted model, what next; `resume_mode` only for a resumable run, and `worker` (`pid`, `state`: `running`, `starting`, `stale` or `exited`, `heartbeat_age_s`, `log`) only when the run has a worker or is training. | `files.state` (`state.json`), `files.epochs` and `files.result` (`null` until the run has one). |
| `evaluate` | `judge_model` and `compare_model` (`null` without one), `headline` (the judge pass rate of `base` and `tuned`, and `compare` with a compare target, their `delta` with its `ci95`, the paired `wins`, `regressions` and `ties`, `tuned_minus_compare` with a compare target, with the cases it `skipped`, with `--evaluators` the headline `evaluator` and each one's rates and deltas under `evaluators`, and the numbers of `cases` and `trajectories`), each metric's rate per target, the errors per target, `reused` (the targets that took answers or verdicts from the store, with how many), the publication and the caveat. | `files.summary` (`summary.json`: per target the judge's passes, every metric's `n`, `passed` and `rate`, the agreement per trajectory, tokens and errors by stage; the paired counts; the bootstrap's resamples and seed) and `files.results` (`results.jsonl`, one line per judged pair). |

On Pup 2.0 and later, JSON and YAML output wrap the result in Pup's envelope,
`{"status", "data", "metadata"}`, with the result as `data`. With `--jq`, the filtered value
becomes `data`. Earlier Pup releases add the envelope only to JSON output, and only under a
coding agent.

Pup's flags go before `finetune`, as in `pup --output yaml finetune status`. To print the
result alone, add `--no-envelope`, as in
`pup --no-envelope --jq .phase finetune status`, which prints only the run's phase.
`--no-envelope` is a Pup 2.x flag, so on earlier releases run the command without it.

Under pup, `--output table` (or `csv`) without a `--jq` filter shows the rows of a list:
`models list` one row per model, `auth status` per provider, `doctor` per check and
`dataset sample` per sampled row.
