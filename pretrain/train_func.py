import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import Tokenizer


def train_model_simple(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    device: torch.DeviceObjType,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: Tokenizer,
) -> list[torch.Tensor, torch.Tensor, list[int]]:
    pass
