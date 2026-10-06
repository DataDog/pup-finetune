# Test fixtures

Every fixture in this directory is **synthetic**. The conversations, tool calls, tool results,
trace and span IDs, app names, model names, places, SKUs and ticket numbers were written for
these tests. None of it comes from real traces, real users or customer data, and none of it
should: never commit real trace content here.

## `rows/`

Canonical training rows (`pup_finetune.data.schema.TrainingRow`), one JSON object per line.

- `tool_use.jsonl` - valid rows: multi-turn conversations with system prompts, parallel tool
  calls, tool results (including tool errors) and final answers, across a few made-up domains
  (weather, calendar, inventory, support tickets, unit conversion). Some rows carry
  `reasoning_content`, one has an assistant turn with `weight: 0`, two share a trace (a
  sub-agent row), and a few have no `source`, as imported rows may.
- `invalid.jsonl` - one invalid row per `RowErrorCode`, in the order listed in
  `invalid.expected.json`, which records each row's line, id and expected code.
