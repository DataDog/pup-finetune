# How to evaluate and compare models

Read this guide once you have run a first `pup finetune evaluate`. It covers choosing and
calibrating the judge, reading why cases fail, reusing base answers, comparing with another
model and adding your own evaluators. The [README](../README.md#evaluate-the-tuned-model)
shows a first evaluation. [`evaluate`](commands.md#pup-finetune-evaluate) and
[`benchmark`](commands.md#pup-finetune-benchmark) list every flag.

In the examples, replace `<RUN_ID>` with the run id that `train` prints or `pup finetune runs`
lists.

## Choose the judge

LLM judges grade `evaluate` and `benchmark`, and they label rows in `dataset triage`. Without
`--judge-model` or `--judges`, the judges depend on the credentials you have. An Anthropic
credential picks Anthropic judges. Without one, an OpenAI credential picks OpenAI judges, and
otherwise Fireworks models judge. With both an Anthropic and an OpenAI credential, the triage
council has one Anthropic judge and one OpenAI judge, so triage sends each row to both
providers.

The [`evaluate`](commands.md#pup-finetune-evaluate) and
[`dataset triage`](commands.md#pup-finetune-dataset-triage) references name each default
model. To choose the judge yourself, name it:

```sh
# Grade with the judge you name (preview)
pup finetune evaluate <RUN_ID> --judge-model anthropic/claude-opus-5-5
```

The preview names the judge and where the choice came from, in `judge_source`. An evaluation
directory keeps its first judge, so a different judge needs a new `--out-dir`. A
`dataset triage` council takes the same names in `--judges`, and it can mix providers.

Anthropic judges need the `anthropic` extra. Without an API key, they sign in the way the
Anthropic SDK does, for example, with `ANTHROPIC_AUTH_TOKEN` or `ant auth login`.
`pup finetune auth status` reports that sign-in as `anthropic-sdk`.

Before it offers `--confirm`, the preview looks for a key for each model that still has
calls to make: the judge, the compare model and each evaluator's judge. If one is missing, the
preview fails and says how to sign in. With `PUP_READ_ONLY` set, it warns instead. A rerun
with nothing left to judge needs no judge key. A `dataset triage` preview fails on a missing key
even with `PUP_READ_ONLY` set, because read-only mode still allows triage's `--confirm`.

## Calibrate the judge

Each new evaluation directory calibrates the judge before it grades anything. The judge grades
a few recorded answers and two corrupted copies of each, and it must pass every recorded answer
and fail every corrupted copy. Each judge evaluator with `examples` also calibrates, with one
call per graded example. The preview counts the judge's calls in `calibration_calls`, and each
evaluator's calls in the `calibration_calls` of its entry under `evaluators`.

Calibration runs in every new `--out-dir`, whether or not the run reuses base answers. Once it
has passed in a directory, later runs there skip it at no cost. `benchmark` calibrates its
judge the same way. To skip calibration, pass `--skip-calibration`. After `train --evaluate`,
the flag is `--eval-skip-calibration`.

If calibration fails, the evaluation stops with `calibration_failed`, exits `1` and replays no
test case. A compare model that the same run adds is checked and kept before calibration, so
it may already have made its one probe call. When the judge fails and the directory holds no
answers or results yet, rerun with another `--judge-model`. Otherwise, use a new `--out-dir`.
To evaluate anyway, rerun with `--skip-calibration`.

## See why cases fail

On a terminal, `evaluate` and `benchmark` print each failed case and its cause on stderr. The
cause is an answer cut off at `--max-tokens`, an answer that could not be parsed, or the
judge's reason. Off a terminal, only the first failure of each cause prints, and the progress
line counts the rest. To see every failure, set `PUP_FINETUNE_PROGRESS=steps` on the command
that the preview's `next_command` gives.

## Reuse base answers across runs

`evaluate`, `train --evaluate` and `benchmark` store the base model's answers and their
verdicts in the dataset's `replay_cache`. A later run of the same dataset reuses a base answer
only when it was made with the same provider, base model, renderer, context length and
`--max-tokens`. It reuses the answer's verdict only when the judge is also the same.

When every base answer a run needs is reused, no base sampler starts. A reused answer that the
same judge already judged costs nothing. With another judge, each reused answer costs one
judge call. In `evaluate`, you also pay for the tuned model's answers and their judge calls.
`benchmark` has no tuned model. Any base answer that is not reused is made again and billed.
For the calls calibration adds, see [Calibrate the judge](#calibrate-the-judge).

Before you confirm, read the preview's `reuse`. It shows what the run reuses and from which
runs. Runs that reuse the same base answers have deltas that compare directly, but they share
the noise of those answers. To make the base answers again, add `--no-reuse`. Deleting
`datasets/my-agent/replay_cache` is safe. It only costs new answers and their judge calls.

## Compare with the model your app runs today

```sh
# Replay a third model and grade it with the same judge (preview)
pup finetune evaluate <RUN_ID> --compare-model anthropic/claude-opus-5-5
```

`--compare-model` replays the same cases with a third model and grades it with the same judge.
The compare model must differ from the judge. The result adds `headline.tuned_minus_compare`,
and publishing adds a third experiment. Added to a complete evaluation, the compare model
reuses the base and tuned results, so you pay only for its answers and their judge calls.

Every answer comes from the model you name. Anthropic can answer a refused request with
another model on its side. `pup-finetune` asks for that fallback for judges, and turns it off
for the compare model.

One short call first checks that the compare model answers. A mistyped model or a rejected
key fails at that call. After it passes, the evaluation directory keeps the compare model. A
rerun without the flag continues it, and another compare model needs a new `--out-dir`.

Some APIs refuse certain conversations every time. Anthropic refuses one that starts or ends
on an assistant turn, for example. `pup-finetune` never sends those cases. The preview counts
them by reason in `compare.skipped_by_reason` and warns, and they count in none of the compare
model's rates. If no case is left to send, the preview refuses the compare model.
[The evaluation directory](data-format.md#the-evaluation-directory) lists every reason.

## Add your own evaluators

To score answers with your own LLM judges and Python checks, write a TOML spec and pass it
with `--evaluators`:

```sh
# Score the answers with your own evaluators too (preview)
pup finetune evaluate <RUN_ID> --evaluators evals.toml
```

A spec names each evaluator and its type. A `judge` grades with your rubric, and a `check`
runs a Python function:

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

A check is a plain function of the case and the generated answer. It returns `True`, `False`,
`None` when it does not apply, or `(bool, reason)`:

```python
# In checks.py, a check that fails any answer showing digits
def no_card_numbers(case, candidate):
    if any(char.isdigit() for char in candidate.get("content") or ""):
        return False, "the answer shows digits"
    return True
```

Checks are Python you wrote, and they run in this process with your privileges. The preview
runs each check on every recorded answer, free, so a broken check fails before anything is
paid. Added to a complete evaluation, evaluators score the stored answers, so only their
judges' calls are paid. `benchmark --evaluators` works the same way.
[Evaluator specs and checks](data-format.md#evaluator-specs-and-checks) has the full format.

### Calibrate a judge evaluator

Give a judge evaluator `examples`, a file of answers you graded, with at least one pass and one
fail. The evaluation calibrates the evaluator on them before it scores anything. A judge
evaluator without examples is not calibrated, and the preview and the result warn.

If the evaluator fails calibration, the evaluation stops with `calibration_failed`, exits `1`
and replays no test case. The directory keeps an evaluator only after it passes calibration,
so one that fails on the run that adds it is not kept. Fix its rubric or its examples, or give
it another model, then rerun. To evaluate anyway, rerun with `--skip-calibration`, which also
skips the built-in judge's calibration.

### Choose the headline

The spec's `headline` picks what the result leads with. It is `judge` by default, or the name
of one evaluator. With `headline = "all"`, a case passes when the judge passes it and every
evaluator passes or does not apply. A case where an evaluator raised an error is left out of
`all`.

### Change an evaluator

An evaluation directory never drops an evaluator, and it keeps each one as it was first added.
For a check, it pins the hash of the whole file. To change an evaluator, leave its entry in the
spec as it is, and leave a check's file unchanged too. Add the new version as another entry
under a new name, such as `polite_v2`, and put a changed check in a new file. The new evaluator
scores the stored answers, so only its judge's calls are paid. To start over instead, pass a
new `--out-dir`.

A judge evaluator's examples are not part of what the directory keeps, so you can edit them
under the same name. These rules apply once the directory keeps an evaluator. One that failed
calibration on the run that added it was never kept, so you can change any part of it.

A rerun runs the same pinned file, so a check that raised raises again, and `next_command` does
not offer to retry it. The result's hint names the check. Fix the check in a new file under a
new name. `evaluator_errors` in the result counts each evaluator's errors and shows the first
one.

### Read evaluators in the experiments

Each evaluator is one pass-or-fail metric per case in the published experiments. An evaluator
that you add after the experiments were published is missing from them, and the result warns.
To publish experiments with every evaluator, evaluate into a new `--out-dir`.
