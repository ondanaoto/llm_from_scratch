import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader.utils import create_dataloader_v1
from gpt import GPT_CONFIG_124M_SHORT_CONTEXT
from utils import get_raw_text


def create_train_val_dataloader() -> tuple[DataLoader, DataLoader]:
    batch_size = 2
    train_ratio = 0.9
    text = get_raw_text()
    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    train_dataloader = create_dataloader_v1(
        train_text,
        batch_size=batch_size,
        max_length=GPT_CONFIG_124M_SHORT_CONTEXT["context_length"],
        stride=GPT_CONFIG_124M_SHORT_CONTEXT["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = create_dataloader_v1(
        val_text,
        batch_size=batch_size,
        max_length=GPT_CONFIG_124M_SHORT_CONTEXT["context_length"],
        stride=GPT_CONFIG_124M_SHORT_CONTEXT["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    return train_dataloader, val_dataloader


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.DeviceObjType,
) -> torch.Tensor:
    """1つのbatchに対するlossを計算する"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits: torch.Tensor = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.DeviceObjType,
    num_batches: int | None = None,
) -> float:
    """データローダーのbatchに対する損失を算出する
    訓練の評価時に用いる想定なので，勾配は気にしない想定でreturnもfloat．

    Args:
        data_loader (DataLoader): データローダー
        model (nn.Module): GPTモデル
        device (torch.DeviceObjType): デバイス
        num_batches (int | None, optional): 損失の計算の対象とするバッチ数.
        Noneなら全てを対象とする．Defaults to None.

    Returns:
        float: _description_
    """
    if len(data_loader) == 0:
        return float("nan")

    # num_batchesの設定
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    # num_batch分のbatchの損失の平均を算出しreturnする
    total_loss = 0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches
