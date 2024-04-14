import os
import typing
from dataclasses import dataclass

import torch
import numpy as np
import numpy.typing as npt


@dataclass
class OptimizerArgs:
    beta1: float
    beta2: float
    weight_decay: float
    max_lr: float
    min_lr: float
    warmup_iters: int
    cosine_cycle_iters: int


@dataclass
class ModelArgs:
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    attn_pdrop: float | None = None
    residual_pdrop: float | None = None


@dataclass
class TrainerArgs:
    dataset_path: str


@dataclass
class TokenizerArgs:
    tokenizer_path: str


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    N = len(dataset)
    start_indices = np.random.randint(0, N - context_length, size=(batch_size,))
    values = []
    targets = []
    for idx in start_indices:
        values.append(dataset[idx : idx + context_length].tolist())
        targets.append(dataset[idx + 1 : idx + context_length + 1].tolist())
    return torch.tensor(values, device=device), torch.tensor(targets, device=device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(out, "optimizer.pt"))
    torch.save(iteration, os.path.join(out, "iteration.pt"))


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    model.load_state_dict(torch.load(os.path.join(src, "model.pt")))
    optimizer.load_state_dict(torch.load(os.path.join(src, "optimizer.pt")))
    return torch.load(os.path.join(src, "iteration.pt"))
