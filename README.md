# nanoCLM

Learn to score a decision against its context, with a small, readable
[Contrastive Language Model](https://contrastive-lm.notion.site/).

The first experiment is **BANKING77 intent matching**: given a banking query,
choose one of 77 intents. It runs on CPU and trains small projection heads on a
frozen text encoder. It introduces CLM's state–action matching, with a
reproducible evaluation of what that training actually changes.

## Run the experiment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv run --locked prepare_banking77.py
uv run --locked nanoclm.py
uv run --locked evaluate.py
```

uv sets up Python 3.12 and the locked dependencies. The first preparation
downloads the public dataset; the first training run downloads MiniLM. Subsequent
runs reuse those downloads. Dataset files live in `data/`, checkpoints and reports
in `outputs/`; both directories are ignored by Git.

Open `outputs/evaluation.md` for the readable report. Its sibling JSON contains
every prediction, test row ID, candidate text, configuration, and provenance hash.
The default checkpoint is `outputs/nanoclm.pt`. Rerunning training or evaluation
replaces its configured output, so change the paths to keep an experiment.

Each script has a typed `Config` at the top: edit settings there. Type annotations
help editors and type checkers; Python does not enforce them automatically.
There is no argument parser or configuration framework.

## What goes through the model?

For example, a query about when a card will arrive is the **state**. The text
`card arrival` is the matching **action**. Here “action” means an intent to select;
the model does not actually operate a bank account.

```text
state text  → frozen MiniLM → state head  → normalized vector ┐
                                                           ├→ dot product → score
action text → frozen MiniLM → action head → normalized vector┘
```

An **embedding** is a vector representing text. MiniLM already supplies useful
embeddings. We freeze its weights and encode our texts once, so repeated training
steps update only the two small heads and a learned score scale.

Each head is `Linear → GELU → Linear`: a learned transformation, a nonlinearity,
and a projection into a smaller space. Separate heads let the context and the
decision play different roles. The defaults have 53,441 trainable parameters.

Normalization gives every projected vector length one. Their dot product is
then cosine similarity: a measure of alignment. The model multiplies it by a
positive, learned **inverse temperature**. Higher values sharpen the softmax
used during training; they do not change which candidate has the highest score.
The scale is capped at 100 for numerical stability.

## What does training teach?

A batch contains matched pairs. With two examples, its score matrix looks like:

| | card arrival | cash withdrawal charge |
| --- | --- | --- |
| query about a card arriving | matching pair | negative example |
| query about an ATM fee | negative example | matching pair |

The diagonal contains the known matches. Cross-entropy encourages each row to
prefer its diagonal entry over the other actions. We also do this in the reverse
direction, retrieving a state from an action, and average the two losses. This
is **bidirectional InfoNCE**: the training signal is “prefer the observed match
over these alternatives.”

One intent can have many valid queries. Sampling one query per distinct intent
keeps equivalent actions from becoming negatives against each other within a
batch. The sampler balances intents, not their natural traffic frequencies.
Unrelated batch members are still assumed to be incorrect matches; that
assumption needs revisiting for data with multiple valid answers.

AdamW updates the heads. `config.weight_decay` shrinks weights during updates, gradient
clipping limits unusually large updates, and the learning rate warms up before
decaying. These are editable training choices, not evidence that this recipe is
optimal.

Validation checks **every validation query against all 77 intents** and chooses
the checkpoint with the lowest classification cross-entropy. It includes step
zero: if training never improves validation loss, the saved model remains
untrained. Validation is not the bidirectional batch loss: many queries share
the same intent, so its target matrix is not one-to-one.

## What data are we using?

[BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets#banking) contains
10,003 training and 3,080 test banking queries across 77 intents. It was introduced
by Casanueva et al. in
[Efficient Intent Detection with Dual Sentence Encoders (2020)](https://arxiv.org/abs/2003.04807)
and is released under [CC BY 4.0](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE).

Our preparation pins the source commit and checks file hashes. It samples:

| Split | Per intent | Total | Source |
| --- | ---: | ---: | --- |
| Train | 10 | 770 | Official training data |
| Validation | 5 | 385 | Other official training rows |
| Test | 5 | 385 | Official test data |

Candidate descriptions are simply the original intent names with underscores
replaced by spaces. No descriptions are designed from test examples. The
prepared file includes those candidates, sampling settings, and zero-based CSV
row IDs. Queries keep their original text.

For duplicate detection only, we casefold and collapse whitespace. We retain the
first occurrence, process official training data first, and exclude any test
text seen there. This excludes 4 training rows and 8 test rows in the pinned
source (7 train/test overlaps and 1 repeated test query). Semantic paraphrases
are not detected by this check. Changing these rules changes the experiment.

This is a **BANKING77 sample result**, not a full-benchmark score. Test queries
are unseen during head training, but the intents are shared. It does not test
unseen-intent generalization, actual banking outcomes, or whether the pretrained
encoder encountered these public texts in its own training.

## How do we know whether training helped?

`evaluate.py` loads the saved checkpoint, verifies the prepared-data hash, and
uses its encoder revision and input-length setting. Every method receives the
same test queries and the same 77 candidate intents:

| Method | What it measures |
| --- | --- |
| Frozen cosine | Match raw query embeddings to candidate-text embeddings. |
| Untrained heads | Recreate the heads at their recorded initialization seed. |
| Trained nanoCLM | Load the validation-selected checkpoint. |
| Nearest centroid | Average training-query embeddings for each intent, normalize, then choose the nearest average. |

Nearest centroid uses the same labelled training queries, with no learned head.
It asks whether contrastive training adds value beyond a simple use of the
encoder. Beating untrained heads alone does not answer that question.

**Accuracy** is the fraction of queries with the correct top choice. **Macro-F1**
averages each intent's F1 score, so every intent has equal weight. F1 accounts for
both missed queries and queries incorrectly assigned to an intent. Errors are
shown in fixed dataset order, and all predictions are available in JSON.

The report also gives paired accuracy differences and 95% bootstrap intervals.
It resamples queries *within each intent*, evaluating both methods on the same
resampled queries. An interval crossing zero means the sample does not clearly
separate them under this procedure. These intervals condition on the fitted
models; they do not establish performance on a different domain or on unseen
training seeds.

Runtime includes shared embedding work once. Per-method scoring timings exclude
encoding and are single-run diagnostics, not production latency measurements.
A softmax over matching scores is **not automatically calibrated confidence**;
this experiment does not claim to measure trustworthy probabilities.

For another checkpoint, edit `evaluate.Config.checkpoints` and `output_path`.
Several checkpoints in one report must share the same training configuration
(apart from seeds and paths). Compare different recipes through separate reports.
The evaluator also checks the trainer source hash: evaluate and save your report
before editing the trainer, then retrain and evaluate the new version.

### Repeat across training seeds

One run is the quick default. For a stronger comparison, this trains five seeds
on the same fixed data, reuses embeddings, and evaluates only after all training
runs finish:

```bash
uv run --locked python - <<'PY'
from dataclasses import replace
from pathlib import Path
import torch
import nanoclm
import evaluate

config = nanoclm.Config()
torch.set_num_threads(config.num_threads)
pairs = nanoclm.load_pairs(config)
train_data, val_data = nanoclm.prepare_data(pairs, config)
paths = []
for seed in (42, 43, 44, 45, 46):
    path = Path(f"outputs/seed-{seed}.pt")
    run = replace(config, seed=seed, checkpoint_path=path)
    torch.manual_seed(seed)
    model = nanoclm.NanoCLM(train_data.states.shape[1], run.hidden_dim, run.projection_dim).to(run.device)
    nanoclm.train(model, train_data, val_data, run)
    paths.append(path)
evaluate.evaluate(evaluate.Config(checkpoints=tuple(paths), output_path=Path("outputs/five-seeds.json")))
PY
```

The report separates variation across seeds from test-sample uncertainty. Five
runs reuse the same 385 queries; they do not create 1,925 independent test cases.
Its paired intervals use the mean correctness difference across the fitted
seeds for each query. The classifier is not an ensemble: per-run scores remain
separate. An unchanged encoder gives the same centroid baseline across seeds.

Choose changes using validation, then evaluate the final experiment. Repeatedly
choosing changes based on test results turns the test set into development data.

## Things to try

Change one thing, preserve the split and candidate texts, and compare reports:

- Increase `config.batch_size`: more competing actions per update.
- Change `config.projection_dim` or `config.hidden_dim`: change head capacity.
- Change `config.learning_rate` or `config.weight_decay`: change optimization.

Changing the data budget or encoder answers a different question; label that
comparison accordingly. A checkpoint only works with the encoder, pooling, input
length, and head architecture it was trained with. Input beyond
`config.max_seq_length` is truncated. Raising the limit alone does not give an
encoder longer-context capability.

For another fixed-intent dataset, keep the prepared JSON structure. Each pair
contains `id`, `state`, and `action`; `candidates` lists the action texts and
`metadata` identifies the source. The training code reads only train/validation
pairs. The evaluator requires all three splits, checks repeated texts/IDs, and
requires each split to cover the declared candidates.

## How this connects to agent trajectories

The [original CLM work](https://github.com/Contrastive-LM/CLM) uses the same
matching idea with questions/answers and agent contexts/actions. Our example
uses frozen MiniLM with mean pooling and small heads, rather than the reference
Qwen3-8B encoder, last-token pooling, larger heads, and large-scale training data.

For a trajectory, the state can contain execution history and the action is a
decision at that step. The released
[evaluator](https://github.com/Contrastive-LM/CLM/blob/main/evaluation/bon_eval.py)
scores individual steps, averages scores over a final window, and selects among
candidate trajectories. Evaluation then asks whether the selected solution
actually passed, compared with random selection and an oracle.

That requires suitable traces, history encoding, task-disjoint splits, and
recorded outcomes. It is the next educational example; BANKING77 does not
establish trajectory-judging capability. Offline selection quality and the
effect of choosing actions inside a running agent are separate experiments.

## Checks and measured results

```bash
uv run --locked python -m unittest discover -s tests -v
```

These focused checks cover metric arithmetic, paired intervals, split isolation,
source deduplication, batching, and all-candidate validation. The end-to-end
experiment and its limitations are recorded in [RESULTS.md](RESULTS.md).
