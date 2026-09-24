# nanoclm

A small, single-file trainer for Contrastive Language Models, inspired by
[CLM](https://contrastive-lm.notion.site/) and its
[reference implementation](https://github.com/Contrastive-LM/CLM).

With uv installed, run:

```bash
uv run --locked nanoclm.py
```

uv sets up Python 3.12 and the locked dependencies. The defaults run on CPU;
the first run downloads the encoder. Edit `Config` at the top of `nanoclm.py`
to change the settings.

The script encodes texts once with frozen MiniLM, then trains two small projection
heads and a score scale using bidirectional InfoNCE. Training uses AdamW, warmup
and cosine decay, and gradient clipping. It prints validation metrics and saves
the best validation checkpoint to `outputs/nanoclm.pt`, replacing it on each run.

`data.json` supplies `train` and `val` lists of matched text pairs:

```json
{"state": "My parcel has not arrived.", "action": "Send the request to the Maple team."}
```

States must be unique within and across splits. Each split needs at least
`config.batch_size` distinct actions. Batches sample one state per distinct action;
other actions in the batch are assumed incorrect. Validation uses fixed sampled
batches. All embeddings stay in memory.

The supplied 48 training and 16 validation examples are a small starting dataset.
This trains the heads only, uses MiniLM mean pooling, and does not reproduce the
reference CLM's Qwen encoder, training scale, or benchmark results.
