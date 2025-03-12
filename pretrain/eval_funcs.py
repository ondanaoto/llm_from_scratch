import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gpt import GPTModel
from gpt.utils import generate_text_simple
from tokenizer import Tokenizer
from tokenizer.utils import text_to_token_ids, token_ids_to_text

from .utils import calc_loss_loader


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.DeviceObjType,
    eval_iter: int,
) -> tuple[float, float]:
    # 評価モードにする
    model.eval()

    train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    # 訓練モードに戻す
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(
    model: GPTModel,
    tokenizer: Tokenizer,
    device: torch.DeviceObjType,
    start_context: str,
) -> None:
    """次単語予測をさせてどんな文章が出来上がるかを見て評価する

    Args:
        model (GPTModel): GPTモデル
        tokenizer (Tokenizer): トークナイザ
        device (torch.DeviceObjType): デバイス
        start_context (str): 次単語予測のための最初のテキスト
    """
    model.eval()
    ids = text_to_token_ids(start_context, tokenizer).to(device)
    context_size = model.pos_emb.weight.shape[0]

    with torch.no_grad():
        decoded_ids = generate_text_simple(
            model, ids, max_new_tokens=50, context_size=context_size
        )

    decoded_text: str = token_ids_to_text(decoded_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
