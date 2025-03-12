import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import Tokenizer

from .eval_funcs import evaluate_model, generate_and_print_sample
from .utils import calc_loss_batch


def train_model_simple(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.DeviceObjType,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: Tokenizer,
) -> list[torch.Tensor, torch.Tensor, list[int]]:
    """訓練を行う

    Args:
        model (nn.Module): 訓練対象のモデル
        train_loader (DataLoader): 訓練のデータセットのデータローダー
        val_loader (DataLoader): 評価のデータセットのデータローダー
        optimizer (torch.optim.Optimizer): 訓練時に使用するオプティマイザー
        device (torch.DeviceObjType): 訓練を実行するデバイス
        num_epochs (int): 訓練のエポック数
        eval_freq (int): 訓練の何回に1回評価を行うか
        eval_iter (int): 評価時に何バッチ分を評価に用いるか
        start_context (str): 次単語予測をさせて評価するためのサンプル
        tokenizer (Tokenizer): 評価時の次単語予測で用いるトークナイザー

    Returns:
        list[torch.Tensor, torch.Tensor, list[int]]: 最終的な評価結果．
        訓練データによる損失の評価結果, 検証データによる損失の評価結果, 総使用トークン数
    """
    # 評価の記録用に空リストやintを用意する．
    # train_losses, val_lossesは評価時のlossの値の歴史
    # track_token_seenは評価のタイミング時に，訓練で使用してきたトークン数(単調増加)
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        for input_batch, target_batch in train_loader:
            assert type(input_batch) is torch.Tensor
            # 訓練
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            # 記録
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}"
                    f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen
