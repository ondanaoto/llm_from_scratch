import pytest
import torch

from gpt import GPT_CONFIG_124M_SHORT_CONTEXT, GPTModel
from pretrain.utils import calc_loss_loader, create_train_val_dataloader


def test_tools():
    torch.manual_seed(123)
    train_loader, val_loader = create_train_val_dataloader()
    model = GPTModel(GPT_CONFIG_124M_SHORT_CONTEXT)
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader,
            model,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        val_loss = calc_loss_loader(
            val_loader,
            model,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    assert pytest.approx(10.99092928568522, 1e-5) == train_loss
    assert pytest.approx(10.982184410095215, 1e-5) == val_loss
