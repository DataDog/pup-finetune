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

`dataset preview` and `dataset pull` read LLM spans through the Agent Observability Export
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
so every row of a trace lands in the same split.

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
| `settings.json` | The source (`import` or `datadog`) and its settings, such as the span filter and time window, plus the split settings. Frozen on first use: later runs must pass the same ones. |
| `rows.jsonl` | The accepted rows, one canonical row per line. Never rewritten by labeling. |
| `excluded.jsonl` | One `{"code", "detail", "row_id", "trace_id", "span_id", "line_sha256"}` per excluded row (fields that do not apply are left out). |
| `labels.jsonl` | Keep/drop decisions appended by `dataset label` and `dataset triage`: `{"id", "keep", "reason", "by", "at"}`, where `by` is `manual` (recorded with `dataset label`) or `judge` (decided by the `dataset triage` council; triage never relabels a manual row). Older files that say `agent` or `human` read as `manual`. The last label of an id wins, a reason is required to drop, and a dropped row never trains and is never pushed. |
| `splits.json` | The split of every trace and the frozen seed and fractions. A trace never changes split once assigned. |
| `summary.json` | Counts, splits and warnings, rewritten after every change. |
| `raw/` | For a Datadog pull, the Export API pages it read, so a rerun resumes without fetching them again. |
| `judgments.jsonl`, `triage.json` | For `dataset triage`, every judge vote (`{"row_id", "row_sha256", "judge", "criteria_sha256", "keep", "reason", "at"}`, plus `"too_long": true` when the judge refused the row as longer than its context window, which drops it) and the frozen rubric hash, judges and the settings they are called with. |
| `push.json` | The receipt `dataset push` writes (see below). |

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
rows not yet pushed and an interrupted push resumes without adding a row twice. `evaluate`
publishes its Agent Observability experiments on the dataset and version this receipt names.
