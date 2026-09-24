"""Download a pinned BANKING77 release and make a small, repeatable experiment."""

import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True)
class Config:
    output_path: Path = Path(__file__).parent / "data" / "banking77.json"
    seed: int = 42
    train_per_intent: int = 10
    val_per_intent: int = 5
    test_per_intent: int = 5


REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}"
CHECKSUMS = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    "LICENSE": "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
}


def download_sources(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, checksum in CHECKSUMS.items():
        path = directory / name
        if not path.exists():
            location = name if name == "LICENSE" else f"banking_data/{name}"
            with urlopen(f"{SOURCE}/{location}", timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError(f"Unexpected download contents: {name}")
            path.write_bytes(content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Source checksum mismatch: {path}")


def read_unique_rows(path, split, seen):
    """Keep the first occurrence; exclude test texts already in official train."""
    groups = defaultdict(list)
    skipped = 0
    with path.open(newline="", encoding="utf-8") as source:
        for index, row in enumerate(csv.DictReader(source)):
            key = " ".join(row["text"].casefold().split())
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            intent = row["category"]
            groups[intent].append({
                "id": f"{split}:{index}",  # Zero-based data row, excluding CSV header.
                "state": row["text"],
                "action": intent.replace("_", " "),
                "intent": intent,
            })
    return groups, skipped


def prepare(config: Config):
    if min(config.train_per_intent, config.val_per_intent, config.test_per_intent) < 1:
        raise ValueError("Each split needs at least one query per intent")
    directory = config.output_path.parent / "source"
    download_sources(directory)
    seen = set()
    train, skipped_train = read_unique_rows(directory / "train.csv", "train", seen)
    test, skipped_test = read_unique_rows(directory / "test.csv", "test", seen)
    intents = sorted(json.loads((directory / "categories.json").read_text()))
    if set(train) != set(intents) or set(test) != set(intents):
        raise ValueError("Source categories do not agree")

    data = {"train": [], "val": [], "test": []}
    rng = random.Random(config.seed)
    for intent in intents:
        count = config.train_per_intent + config.val_per_intent
        selected = rng.sample(train[intent], count)
        data["train"].extend(selected[:config.train_per_intent])
        data["val"].extend(selected[config.train_per_intent:])
        data["test"].extend(rng.sample(test[intent], config.test_per_intent))
    data["candidates"] = [intent.replace("_", " ") for intent in intents]
    data["metadata"] = {
        "dataset": "BANKING77 sample (not the full benchmark)",
        "source": "https://github.com/PolyAI-LDN/task-specific-datasets",
        "citation": "Casanueva et al. (2020), Efficient Intent Detection with Dual Sentence Encoders",
        "license": "CC BY 4.0",
        "source_revision": REVISION,
        "source_sha256": CHECKSUMS,
        "sampling": {k: v for k, v in asdict(config).items() if k != "output_path"},
        "excluded_rows": {"train": skipped_train, "test": skipped_test},
        "normalization_for_deduplication": "casefold and collapse whitespace; keep train before test",
        "candidate_text": "source intent with underscores replaced by spaces",
    }
    config.output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    counts = ", ".join(f"{len(data[s])} {s}" for s in ("train", "val", "test"))
    print(f"Saved {config.output_path}: {counts}; {len(intents)} candidates")
    print(f"Excluded duplicate/overlapping source rows: train={skipped_train}, test={skipped_test}")
    return data


if __name__ == "__main__":
    prepare(Config())
