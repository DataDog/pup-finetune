# How to train a model

Read this guide once you have run a first `pup finetune train`. It covers capping what a run
spends, decaying the learning rate, stopping and resuming a run, and training in a Baseten
job. The [README](../README.md#train-a-model) shows a first run, and
[`train`](commands.md#pup-finetune-train) lists every flag.

Replace `<PROVIDER>` with `baseten` or `fireworks`, and `<RUN_ID>` with the run id that `train`
prints or `pup finetune runs` lists.

## Cap what a run spends

When a cap stops training, the run keeps the best epoch so far. It still ends `completed`, so
you can evaluate and deploy it.

`--max-train-tokens` is a hard limit. Training never starts a step that would take the trained
tokens over it. Set it to at least the preview's `tokens_per_epoch`. A lower cap lets no epoch
finish, so the run keeps nothing, and the preview warns.

`--max-spend-usd` caps an estimate, not the provider's bill. The guard prices the elapsed time
at the hourly rate you give with `--hourly-rate-usd`:

```sh
# Stop near $45 of a $50 estimate, keeping $5 for the final checkpoint (preview)
pup finetune train --provider <PROVIDER> --dataset datasets/my-agent --model qwen3p8-27b \
  --max-spend-usd 50 --hourly-rate-usd <USD_PER_HOUR>
```

Replace `<USD_PER_HOUR>` with your estimate of what training costs per hour. Your provider may
bill another way, for example per token, so the real bill can differ from the estimate.

The guard checks the estimate before each paid call, such as a training step or a checkpoint
save. Training stops once the estimate reaches `--max-spend-usd` less a reserve, and the reserve
is kept for the final checkpoint. The reserve is `--spend-reserve-fraction` of the cap, 0.1 by
default. The guard cannot stop a call in flight, so a slow call can take the estimate past
`--max-spend-usd`. If the estimate has reached `--max-spend-usd` when training stops, the run
skips the final checkpoint and warns.

With `--evaluate`, the evaluation after training also stops, partial, at `--max-spend-usd`. A
run that a cap or a cancel stopped is not evaluated in its training session. Run
`pup finetune evaluate <RUN_ID>` to evaluate it.

`--max-seq-len` caps the context, and with it the cost. Rows longer than the cap are left out,
and the preview counts them as `exceeds_context`. On Baseten, the cap is at least 1024 tokens.

## Decay the learning rate

The learning rate stays constant unless `--lr-schedule linear` or `cosine` decays it:

```sh
# Warm up over 5% of the steps, then decay over 3 epochs (preview)
pup finetune train --provider <PROVIDER> --dataset datasets/my-agent --model qwen3p8-27b \
  --lr-schedule cosine --warmup-fraction 0.05 --max-epochs 3 --patience 3
```

A schedule decays over the steps of `--max-epochs` epochs, which the preview shows as
`max_steps`. A linear warmup over `--warmup-fraction` of those steps comes first, shown as
`warmup_steps`. The rate never reaches zero.

Early stopping or a cap can end the run before the decay finishes. Training stops after
`--patience` epochs without a validation-loss improvement of at least `--min-delta`. To let
the decay finish, set `--max-epochs` to the length you want, raise `--patience`, and set any
cap above what those epochs need.

## Stop and resume a run

To stop a run, cancel it. The training process finishes its current step, saves a resumable
checkpoint and keeps the best epoch so far. Ctrl-C in a foreground run does the same.

```sh
# Show which process gets the signal (preview), then run the next_command it prints
pup finetune cancel <RUN_ID>
```

An error during training leaves the run `interrupted`, and its hint names the resume command.
`resume` continues an interrupted, failed or cancelled run from its last training checkpoint.
When training already ended, it only retries the promotion.

```sh
# Show where the run resumes (preview), with --detach for a background worker
pup finetune resume <RUN_ID>
```

## Train in a Baseten job

On Baseten, `--remote` runs the training loop in a Baseten CPU job instead of on your machine,
so training goes on while your machine sleeps. The job bills on top of the Loops trainer.

```sh
# Train in a Baseten job (preview), then follow it
pup finetune train --provider baseten --dataset datasets/my-agent --model qwen3p5-9b --remote
pup finetune status --watch
```

Before you confirm, check that you have:

- the `baseten` CLI and `uv` on your `PATH`, and a source checkout of pup-finetune.
- a team secret named `pup_finetune_api_key` that holds a Baseten API key of the same
  workspace as your local key. `--team` names the team, by default your only or default team.

`train --remote` uploads the train and validation rows, their split assignments, the run files
and pup-finetune's source to Baseten storage. The test conversations and the dropped rows stay
on your machine. `--remote` is refused with either `--detach` or `--evaluate`.

`status` reads the job and its logs and copies the run's state into the run directory. Then
`status --watch`, `evaluate` and `deploy` work as for a local run. `cancel` cannot stop the
job, so stop the job and its Loops trainer in the Baseten UI. `resume` is refused until the
job ended, and then it trains on your machine.
