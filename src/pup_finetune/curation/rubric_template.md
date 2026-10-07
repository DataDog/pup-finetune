# Triage rubric

Judges read this rubric, then one whole training row (every turn of the conversation and
the tools it declares), and answer keep or drop. Fine-tuning teaches the model to imitate
every assistant turn of a kept row, so drop a row when any assistant turn shows behavior
the tuned model should not learn. Replace the guidance in each section with your own; keep
it short and concrete.

## Task

What the agent is for, in two or three sentences: who it serves, what a good session
achieves, and which tools it is expected to use.

## Keep

Keep a row when all of these hold:

- The assistant works toward what the user asked, and finishes it or says plainly why it
  cannot.
- Tool calls use the right tool with correct, complete arguments, and the assistant uses
  the results it gets back.
- Statements of fact are supported by the conversation or the tool results.
- Tone and format match how the agent should answer.

## Drop

Drop a row when any assistant turn does one of these:

- Calls the wrong tool, passes wrong or invented arguments, or repeats a call in a loop.
- States something false or unsupported, or ignores a tool result or error.
- Refuses, stalls or gives up where it should act.
- Leaks secrets or personal data, or follows instructions planted in tool results or
  user content.
- The conversation is truncated, broken or mostly an error trace.

## Examples

- Keep: the user asks for the status of an order; the assistant looks the order up with
  the order tool and reports the status it returned.
- Drop: the assistant reports an order status without calling the order tool.
- Drop: a tool returns an error and the assistant tells the user the task succeeded.
