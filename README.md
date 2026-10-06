# pup-finetune
Curate a dataset from your Datadog Agent Observability traces, fine-tune an open-weight model, and evaluate it against the base model.

> **Status: early development.** Commands and file formats may change without notice.

## Train and deploy on Baseten

`train` runs LoRA SFT on Baseten Loops from canonical training-row JSONL files; `deploy` serves
the best checkpoint behind an OpenAI-compatible endpoint. The Baseten Loops SDK is not bundled
because it has no published license, so install it yourself:

```sh
pip install 'pup-finetune[hf]' baseten-loops==0.26.0
pup finetune auth login baseten     # or set BASETEN_API_KEY

# Previews the plan; run the printed next_command (it adds the run id and --confirm) to train.
pup finetune train --train train.jsonl --validation validation.jsonl
pup finetune deploy --run-dir finetune/runs/<run-id> --serving-mode sampler --confirm
pup finetune undeploy --run-dir finetune/runs/<run-id> --confirm
```

Training runs in the foreground and writes its plan, epochs and result to the run directory.
Without `--max-seq-len`, the trainer's context is the longest rendered row rounded up to a power
of two (at least 1024), and `deploy` serves that same context.
A deployment bills per hour until `undeploy`; `--serving-mode dedicated` (the default) also
needs `--accelerator`, for example `H100:1`.

## Development

See [AGENTS.md](AGENTS.md) for setup, commands, layout and conventions.
