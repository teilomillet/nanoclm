"""Compare saved nanoCLM heads and simple baselines on one fixed test sample."""

import hashlib
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F

import nanoclm
from nanoclm import Config as TrainingConfig
from nanoclm import NanoCLM, encode_pairs


@dataclass(frozen=True)
class Config:
    data_path: Path = TrainingConfig().data_path
    checkpoints: tuple[Path, ...] = (TrainingConfig().checkpoint_path,)
    output_path: Path = Path(__file__).parent / "outputs" / "evaluation.json"
    device: str = "cpu"
    num_threads: int = 2
    bootstrap_samples: int = 2000
    seed: int = 1234
    mistake_count: int = 5


def validate_data(data):
    """Protect the evaluation boundary before computing any model scores."""
    candidates = data["candidates"]
    if len(candidates) < 2 or len(set(candidates)) != len(candidates):
        raise ValueError("Candidates must be distinct")
    if any(not isinstance(c, str) or not c.strip() for c in candidates):
        raise ValueError("Candidates must be nonempty text")
    seen_texts, seen_ids = set(), set()
    for split in ("train", "val", "test"):
        rows = data[split]
        if {r["action"] for r in rows} != set(candidates):
            raise ValueError(f"{split} must contain exactly the declared candidate actions")
        for row in rows:
            text = " ".join(row["state"].casefold().split())
            if not text or text in seen_texts or row["id"] in seen_ids:
                raise ValueError("Empty or repeated query/ID within or across splits")
            seen_texts.add(text)
            seen_ids.add(row["id"])


def classification_metrics(predictions, targets, classes):
    # Rows are true classes, columns are predicted classes.
    confusion = torch.bincount(targets * classes + predictions, minlength=classes**2)
    confusion = confusion.reshape(classes, classes).float()
    true_positives = confusion.diag()
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = 2 * true_positives / denominator.clamp_min(1)
    return {
        "correct": int((predictions == targets).sum()),
        "total": len(targets),
        "accuracy": float((predictions == targets).double().mean()),
        "macro_f1": float(f1.mean()),
    }


def paired_interval(differences, targets, config):
    """Resample queries within each intent, preserving our balanced test design.

    Each value is correctness(A) minus correctness(B) on the SAME query.
    For repeated runs, values can be the mean difference across fixed seeds.
    This interval measures test-sample uncertainty, not training-seed uncertainty.
    """
    generator = torch.Generator().manual_seed(config.seed)
    totals = torch.zeros(config.bootstrap_samples, dtype=torch.float64)
    for label in targets.unique():
        values = differences[targets == label].double()
        draws = torch.randint(len(values), (config.bootstrap_samples, len(values)), generator=generator)
        totals += values[draws].sum(1)
    lower, upper = torch.quantile(totals / len(targets), torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {"accuracy_delta": float(differences.double().mean()), "ci95": [float(lower), float(upper)]}


def nearest_centroid(train):
    """A no-optimization classifier: average training embeddings for each intent."""
    centers = torch.stack([train.states[indices].mean(0) for indices in train.state_indices_by_action])
    return F.normalize(centers, dim=-1)


def write_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# BANKING77 sample evaluation", "",
        f"{report['counts']['test']} test queries, {len(report['candidates'])} candidates per query. "
        "This is a sampled experiment, not a full BANKING77 benchmark score.", "",
        f"Dataset SHA-256: `{report['data_sha256']}`", "",
        "| Model | Correct | Accuracy | Macro-F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, result in report["models"].items():
        m = result["metrics"]
        lines.append(f"| {name} | {m['correct']}/{m['total']} | {m['accuracy']:.2%} | {m['macro_f1']:.4f} |")
    lines += ["", "Paired accuracy differences (percentage points; stratified 95% bootstrap intervals):", ""]
    for name, result in report["comparisons"].items():
        low, high = result["ci95"]
        lines.append(f"- {name}: {100 * result['accuracy_delta']:+.2f} pp "
                     f"[{100 * low:+.2f}, {100 * high:+.2f}].")
    repeated = report["repeat_summary"]
    lines += ["", f"Trained accuracy across {repeated['runs']} run(s): mean {repeated['accuracy_mean']:.2%}; "
              f"sample SD {repeated['accuracy_sd']:.2%}. These runs reuse the same test queries.", "",
              "Intervals condition on these fitted models and resample queries within each intent. "
              "They do not cover new intents, dataset shifts, or uncertainty over unseen training seeds.", "",
              f"Evaluation wall time: {report['timing']['evaluation_seconds']:.2f}s; "
              f"embedding time within it: {report['timing']['embedding_seconds']:.2f}s. "
              "Shared embeddings are computed once. Scoring timings in JSON exclude encoding; "
              "these are single-run diagnostics, not a serving latency benchmark.", "",
              "Mistakes below are the first errors in fixed dataset order, not hand-picked examples.", ""]
    for name, result in report["models"].items():
        if not name.startswith("trained:"):
            continue
        lines += [f"## {name}", ""]
        for mistake in result["mistakes"]:
            lines.append(f"- `{mistake['id']}`: {mistake['state']} "
                         f"Expected **{mistake['expected']}**; predicted **{mistake['predicted']}**.")
        lines.append("")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


@torch.no_grad()
def evaluate(config: Config):
    if not config.checkpoints or config.bootstrap_samples < 1:
        raise ValueError("Supply at least one checkpoint and one bootstrap sample")
    torch.set_num_threads(config.num_threads)
    started = time.perf_counter()
    data_bytes = config.data_path.read_bytes()
    data = json.loads(data_bytes)
    validate_data(data)
    digest = hashlib.sha256(data_bytes).hexdigest()
    checkpoints = [torch.load(path, map_location="cpu", weights_only=True) for path in config.checkpoints]
    first = checkpoints[0]
    trainer_digest = hashlib.sha256(Path(nanoclm.__file__).read_bytes()).hexdigest()
    # Aggregate seed variation only within one experiment; compare different recipes
    # using separate reports. Otherwise a mean could hide which change helped.
    ignored = {"seed", "checkpoint_path", "data_path"}
    reference_config = {k: v for k, v in first["training_config"].items() if k not in ignored}
    for checkpoint in checkpoints:
        if checkpoint.get("data_sha256") != digest:
            raise ValueError("Checkpoint dataset mismatch; train on this exact prepared dataset first")
        if checkpoint["trainer_sha256"] != trainer_digest:
            raise ValueError("Trainer source changed; evaluate with its original code or retrain before comparing")
        run_config = {k: v for k, v in checkpoint["training_config"].items() if k not in ignored}
        if run_config != reference_config or checkpoint["model_config"] != first["model_config"]:
            raise ValueError("Evaluate different training configurations in separate reports")
        if (checkpoint["encoder"], checkpoint["encoder_revision"], checkpoint["training_config"]["max_seq_length"]) != (
            first["encoder"], first["encoder_revision"], first["training_config"]["max_seq_length"]
        ):
            raise ValueError("Evaluate different encoders in separate reports; embeddings must match the checkpoint")

    encoder = SentenceTransformer(first["encoder"], revision=first["encoder_revision"],
                                  device=config.device, trust_remote_code=False)
    encoder.eval()
    encoder.requires_grad_(False)
    encoder.max_seq_length = first["training_config"]["max_seq_length"]
    encoding_started = time.perf_counter()
    candidates = data["candidates"]
    train = encode_pairs(encoder, data["train"], candidates)
    test = encode_pairs(encoder, data["test"], candidates)
    encoding_seconds = time.perf_counter() - encoding_started
    targets = test.targets.cpu()
    report = {
        "data_sha256": digest, "dataset": data["metadata"],
        "counts": {s: len(data[s]) for s in ("train", "val", "test")},
        "candidates": candidates,
        "test_ids": [row["id"] for row in data["test"]], "targets": targets.tolist(),
        "encoder": {"name": first["encoder"], "revision": first["encoder_revision"],
                    "max_seq_length": encoder.max_seq_length},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "platform": platform.platform(), "device": config.device, "threads": config.num_threads},
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256(Path(__file__).with_name("uv.lock").read_bytes()).hexdigest(),
        "bootstrap": {"seed": config.seed, "samples": config.bootstrap_samples,
                      "method": "paired query bootstrap stratified by intent; fixed trained models"},
        "models": {}, "comparisons": {},
    }

    def record(name, score, metadata=None):
        scoring_started = time.perf_counter()
        scores = score()
        if not torch.isfinite(scores).all():
            raise ValueError(f"Non-finite scores from {name}; no valid evaluation is possible")
        predictions = scores.argmax(1).cpu()
        elapsed = time.perf_counter() - scoring_started
        mistakes = []
        for row, prediction, target in zip(data["test"], predictions.tolist(), targets.tolist()):
            if prediction != target and len(mistakes) < config.mistake_count:
                mistakes.append({"id": row["id"], "state": row["state"],
                                 "expected": candidates[target], "predicted": candidates[prediction]})
        report["models"][name] = {
            "metrics": classification_metrics(predictions, targets, len(candidates)),
            "predictions": predictions.tolist(), "mistakes": mistakes,
            "scoring_seconds": elapsed, **(metadata or {}),
        }
        return (predictions == targets).double()

    frozen = record("frozen cosine", lambda: test.states @ test.actions.T)
    centers = nearest_centroid(train)
    centroid = record("nearest centroid", lambda: test.states @ centers.T)
    trained_correct, untrained_correct = [], []
    for index, (path, checkpoint) in enumerate(zip(config.checkpoints, checkpoints)):
        seed = checkpoint["training_config"]["seed"]
        name = f"{index + 1} (seed {seed})"
        torch.manual_seed(seed)
        model = NanoCLM(**checkpoint["model_config"]).to(config.device).eval()
        untrained = record(f"untrained:{name}", lambda: model(test.states, test.actions))
        model.load_state_dict(checkpoint["model"])
        trained = record(f"trained:{name}", lambda: model(test.states, test.actions), {
            "checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "training_config": checkpoint["training_config"], "model_config": checkpoint["model_config"],
            "trainer_sha256": checkpoint["trainer_sha256"],
            "selected_step": checkpoint["step"], "val_loss": checkpoint["val_loss"],
        })
        trained_correct.append(trained)
        untrained_correct.append(untrained)
        metrics = report["models"][f"trained:{name}"]["metrics"]
        print(f"{path.name}: accuracy {metrics['accuracy']:.2%}, macro-F1 {metrics['macro_f1']:.4f}")

    mean_trained = torch.stack(trained_correct).mean(0)
    mean_untrained = torch.stack(untrained_correct).mean(0)
    for name, baseline in (("trained minus untrained", mean_untrained),
                           ("trained minus frozen cosine", frozen), ("trained minus nearest centroid", centroid)):
        report["comparisons"][name] = paired_interval(mean_trained - baseline, targets, config)
    accuracies = [float(values.mean()) for values in trained_correct]
    report["repeat_summary"] = {
        "runs": len(accuracies), "accuracy_mean": statistics.mean(accuracies),
        "accuracy_sd": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
    }
    report["timing"] = {"embedding_seconds": encoding_seconds, "evaluation_seconds": time.perf_counter() - started}
    write_report(report, config.output_path)
    print(f"Saved {config.output_path} and {config.output_path.with_suffix('.md')}")
    return report


if __name__ == "__main__":
    evaluate(Config())
