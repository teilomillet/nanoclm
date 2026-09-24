# BANKING77 sample: first measured experiment

Training learns the matching task, but this small recipe **does not outperform
nearest-centroid classification on the same frozen encoder**. We retain that
result rather than choose settings based on the test set.

Measured on 2026-09-24. The complete report, including per-query predictions,
source checksums, selected checkpoints, configurations, and environment, is
[results/banking77.json](results/banking77.json). Reproduce it with the five-seed
command in the [README](README.md#repeat-across-training-seeds).

## Protocol

- BANKING77 at commit `57ec275d8078af65b7731c2a98be812d844a6d6b`.
- Sampling seed 42; 10 training, 5 validation, and 5 test queries per intent.
- 770 training / 385 validation / 385 test queries; all 77 candidate intents.
- Validation is drawn only from official training data. Test queries come from
  official test data after duplicate/overlap filtering.
- Frozen `sentence-transformers/all-MiniLM-L6-v2` at revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; mean pooling, 256-token limit.
- Two `384 → 64 → 32` heads; 53,441 trainable parameters including the scale.
- 300 updates, batch size 8, peak learning rate 0.003, weight decay 0.01.
- Head initialization and sampling seeds 42–46. All five validation-selected
  checkpoints were at step 300. No hyperparameters were selected using these
  test results.
- CPU, two PyTorch threads, macOS arm64, Python 3.12.12, PyTorch 2.14.0.

Prepared-data SHA-256:
`fc4242186158118ea474f8ee471a7e73db4dcb9aeb1398243b8ea5a6f44f7406`.

## Results

| Method | Test accuracy | Macro-F1 |
| --- | ---: | ---: |
| Frozen cosine matching | 60.00% (231/385) | 0.5829 |
| Nearest centroid from training queries | **82.08% (316/385)** | **0.8172** |
| nanoCLM, seed 42 | 64.42% (248/385) | 0.6353 |
| nanoCLM, seed 43 | 62.60% (241/385) | 0.6095 |
| nanoCLM, seed 44 | 63.90% (246/385) | 0.6295 |
| nanoCLM, seed 45 | 67.01% (258/385) | 0.6591 |
| nanoCLM, seed 46 | 58.70% (226/385) | 0.5697 |

The untrained heads score 4, 5, 7, 4, and 5 correct out of 385, respectively.
Mean trained accuracy is **63.32%**, with a **3.04 percentage-point sample standard
deviation across seeds**. This is not an ensemble result.

Paired differences use mean correctness across the five fitted seeds for each
query. The 95% intervals resample queries within each intent, using 2,000 bootstrap
replicates and seed 1234:

| Comparison | Accuracy difference | 95% interval |
| --- | ---: | ---: |
| Trained minus untrained | +62.03 pp | [+59.17, +64.94] |
| Trained minus frozen cosine | +3.32 pp | [−0.36, +7.12] |
| Trained minus nearest centroid | −18.75 pp | [−21.82, −15.58] |

The gain over random heads is clear. The interval for the gain over frozen cosine
crosses zero. The centroid baseline is substantially stronger in this experiment.
This does not show that contrastive training cannot improve: it shows that these
settings and this data budget have not established that improvement.

## What failed?

The JSON records the first five mistakes per trained run in fixed test order,
plus all predictions. For example, seed 42 confuses a delayed refund with a
request to initiate one; it also maps an Apple Watch top-up question to topping
up by cash or cheque. These are examples of observed errors, not a diagnosed
explanation of why training underperforms.

Inspecting them is a starting point for a hypothesis. Test that hypothesis using
validation or a new held-out experiment, rather than adjusting this test's labels
or candidate descriptions to improve its score.

## Runtime and verification

The final five-seed command took **16.04 seconds wall time** here with the dataset
and encoder already cached, including process startup, encoding, five training
runs, and evaluation. First downloads and initial environment setup are excluded.
This is a local observation, not a speed guarantee or serving benchmark. JSON
contains separate evaluation and embedding timings.

- Downloading into a fresh temporary directory reproduced the prepared JSON
  byte-for-byte and passed all pinned source checksums.
- All 11 focused checks passed, including duplicate detection, split isolation,
  corrupted-source rejection, mismatched-checkpoint rejection, hand-computed
  metrics, and all-candidate validation.
- Scikit-learn independently reproduced accuracy and macro-F1 from every model's
  saved predictions.
- The standalone default workflow and the five-seed workflow produced identical
  seed-42 predictions. Repeating five-seed training reproduced the reported scores.

## Limits

This is a **385-query BANKING77 sample**, not a full-benchmark score and not a
comparison with published Jev or Laya scores under different protocols. Five seeds
reuse the same test examples. The bootstrap intervals condition on these fitted
models; training-seed variation is reported separately. With only five test
queries per intent, finer per-intent conclusions would be weak.

The intents are known during training. Encoder pretraining contamination is not
established or ruled out. Probability calibration, unseen intents, business
outcomes, and trajectory verification are not evaluated here.

Dataset: Casanueva et al. (2020),
[Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807),
[BANKING77 source](https://github.com/PolyAI-LDN/task-specific-datasets), CC BY 4.0.
Our sample removes repeated/overlapping texts, selects the recorded rows, and
converts intent underscores to spaces. The source license is downloaded alongside
the data.
