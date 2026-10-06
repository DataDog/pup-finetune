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

## `llmobs/`

Raw Agent Observability Export API responses (`POST v2/llm-obs/v1/spans/events/search`), as
`pup api` prints them: `{data: [{id, type: "span", attributes}], meta: {page: {after}, status,
...}}`. They are synthetic, written to mirror the shapes real LLM spans take.
`search-page-1.json` links to `search-page-2.json` through `meta.page.after`; the second page
has no cursor, so it is the last.

- `synthetic-trace-0101` - a multi-turn, tool-using trip-planner agent. Its four agent LLM
  spans (`synthetic-span-0101` to `-0104`) have inputs that are growing prefixes of the same
  conversation: assistant messages with `tool_calls` (`arguments` as an object, sometimes
  with content too) and `role: "tool"` messages with empty content and `tool_results`. A
  summarizer sub-agent LLM span (`synthetic-span-0111`, another parent and system prompt)
  sits in the same trace.
- `synthetic-span-0201` - an errored span (`status: "error"`).
- `synthetic-span-0301` - output with a `role: "reasoning"` message before the answer.
- `synthetic-span-0401` - an output message with no `role`.
- `synthetic-span-0501` - a span with no `output`.
- `synthetic-span-0601` - user content as a list of parts, one of them an image.
