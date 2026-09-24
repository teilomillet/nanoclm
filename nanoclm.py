"""Train two CLM projection heads on a frozen text encoder: python nanoclm.py."""

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.nn import functional as F


# Configuration: edit here, then run the file.
@dataclass(frozen=True)
class Config:
    data_path: Path = Path(__file__).parent / "data" / "banking77.json"
    checkpoint_path: Path = Path(__file__).parent / "outputs" / "nanoclm.pt"
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    max_seq_length: int = 256
    device: str = "cpu"
    num_threads: int = 2
    seed: int = 42
    hidden_dim: int = 64
    projection_dim: int = 32
    batch_size: int = 8
    max_steps: int = 300
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    eval_interval: int = 25


@dataclass
class EncodedPairs:
    """Fixed text embeddings, with matching state rows grouped by action."""

    states: torch.Tensor
    actions: torch.Tensor
    state_indices_by_action: list[list[int]]
    targets: torch.Tensor


class NanoCLM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, projection_dim):
        super().__init__()
        self.model_config = {
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "projection_dim": projection_dim,
        }
        self.state_head = projection_head(**self.model_config)
        self.action_head = projection_head(**self.model_config)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, states, actions):
        # Each head learns a different job: describe the context vs. the decision.
        states = F.normalize(self.state_head(states), dim=-1)
        actions = F.normalize(self.action_head(actions), dim=-1)
        # Unit vectors make this cosine similarity. Shape: states x candidates.
        similarity = states @ actions.T
        # Inverse temperature controls how sharply the loss distinguishes matches.
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * similarity


def projection_head(embedding_dim, hidden_dim, projection_dim):
    """Map a frozen text embedding into the space where states match actions."""
    return nn.Sequential(
        nn.Linear(embedding_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, projection_dim),
    )


def contrastive_loss(scores):
    # Pair i is the match in both directions; other batch members are negatives.
    targets = torch.arange(len(scores), device=scores.device)
    state_loss = F.cross_entropy(scores, targets)
    action_loss = F.cross_entropy(scores.T, targets)  # Retrieve states from actions too.
    return (state_loss + action_loss) / 2


def load_pairs(config: Config):
    """Read matched texts and reject empty inputs or repeated states."""
    if config.batch_size < 2:
        raise ValueError("Contrastive training needs at least two pairs per batch")
    if not config.data_path.exists():
        raise FileNotFoundError(f"{config.data_path} is missing; run prepare_banking77.py first")
    data = json.loads(config.data_path.read_text())
    pairs = data["train"] + data["val"]
    for pair in pairs:
        for key in ("state", "action"):
            if not isinstance(pair[key], str) or not pair[key].strip():
                raise ValueError("Every pair needs nonempty state and action strings")
    states = [" ".join(pair["state"].casefold().split()) for pair in pairs]
    if len(states) != len(set(states)):
        raise ValueError("States must be unique within and across train/val splits")
    for split in ("train", "val"):
        if len({pair["action"] for pair in data[split]}) < config.batch_size:
            raise ValueError(f"{split} needs at least config.batch_size distinct actions")
    if {p["action"] for p in data["train"]} != {p["action"] for p in data["val"]}:
        raise ValueError("This example expects the same candidate actions in train and val")
    return data


def prepare_data(pairs, config: Config):
    """Encode both splits once; only the small heads will receive gradients."""
    encoder = SentenceTransformer(
        config.encoder_name, revision=config.encoder_revision, device=config.device,
        trust_remote_code=False,
    )
    encoder.eval()
    encoder.requires_grad_(False)
    encoder.max_seq_length = config.max_seq_length  # Longer text is truncated, not summarized.
    candidates = sorted({p["action"] for p in pairs["train"]})
    train_data = encode_pairs(encoder, pairs["train"], candidates)
    val_data = encode_pairs(encoder, pairs["val"], candidates)
    return train_data, val_data


@torch.no_grad()
def encode_pairs(encoder, pairs, candidates):
    # An action can have several matching states. Keep their row indices together.
    groups = {action: [] for action in candidates}
    for i, pair in enumerate(pairs):
        groups.setdefault(pair["action"], []).append(i)
    states = encoder.encode(
        [pair["state"] for pair in pairs], convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    actions = encoder.encode(
        list(groups), convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    action_ids = {action: i for i, action in enumerate(candidates)}
    targets = torch.tensor([action_ids[p["action"]] for p in pairs], device=states.device)
    return EncodedPairs(states, actions, list(groups.values()), targets)


def sample_batch(data, rng, config: Config):
    # One state per distinct action avoids false negatives from repeated actions.
    action_indices = rng.sample(range(len(data.actions)), config.batch_size)
    state_indices = [rng.choice(data.state_indices_by_action[i]) for i in action_indices]
    return data.states[state_indices], data.actions[action_indices]


@torch.no_grad()
def evaluate(model, data):
    """Validate on every query against every intent, including repeated targets."""
    was_training = model.training
    model.eval()
    scores = model(data.states, data.actions)
    # Validation is classification, not one-to-one batch retrieval.
    loss = F.cross_entropy(scores, data.targets).item()
    accuracy = (scores.argmax(1) == data.targets).float().mean().item()
    model.train(was_training)
    return loss, accuracy


def learning_rate_at(step, config: Config):
    """Warm up for 10% of training, then smoothly decay to 10% of the peak."""
    warmup_steps = max(1, config.max_steps // 10)
    if step < warmup_steps:
        return config.learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, config.max_steps - warmup_steps - 1)
    cosine = (1 + math.cos(math.pi * progress)) / 2
    return config.learning_rate * (0.1 + 0.9 * cosine)


def train(model, train_data, val_data, config: Config):
    rng = random.Random(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    best_loss = math.inf
    model.train()
    print(f"{sum(p.numel() for p in model.parameters()):,} trainable parameters")
    print(f"{len(train_data.states)} train / {len(val_data.states)} val pairs")

    # Step counts completed updates: evaluate before training and at the end too.
    for step in range(config.max_steps + 1):
        if step % config.eval_interval == 0 or step == config.max_steps:
            val_loss, val_accuracy = evaluate(model, val_data)
            print(f"step {step:4d} | val loss {val_loss:.4f} | "
                  f"val accuracy {val_accuracy:.1%} ({len(val_data.actions)} candidates)")
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, step, val_loss, config)
        if step == config.max_steps:
            break

        states, actions = sample_batch(train_data, rng, config)
        scores = model(states, actions)
        loss = contrastive_loss(scores)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, config)
        optimizer.step()


def save_checkpoint(model, step, val_loss, config: Config):
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_config": model.model_config,
        "encoder": config.encoder_name,
        "encoder_revision": config.encoder_revision,
        "step": step,
        "val_loss": val_loss,
        "training_config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "data_sha256": hashlib.sha256(config.data_path.read_bytes()).hexdigest(),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, config.checkpoint_path)
    print(f"Saved best checkpoint at step {step}: {config.checkpoint_path}")


def main(config: Config):
    torch.set_num_threads(config.num_threads)
    pairs = load_pairs(config)
    train_data, val_data = prepare_data(pairs, config)
    embedding_dim = train_data.states.shape[1]
    torch.manual_seed(config.seed)  # Head initialization is independent of encoder loading.
    model = NanoCLM(embedding_dim, config.hidden_dim, config.projection_dim).to(config.device)
    train(model, train_data, val_data, config)


if __name__ == "__main__":
    main(Config())
