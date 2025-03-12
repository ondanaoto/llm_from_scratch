import pytest
import tiktoken
import torch

from gpt import GPT_CONFIG_124M_SHORT_CONTEXT, GPTModel
from pretrain import train_model_simple
from pretrain.utils import create_train_val_dataloader
from view.train_val_loss import plot_losses


@pytest.mark.slow
def test_train_simple_view():
    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M_SHORT_CONTEXT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    train_loader, val_loader = create_train_val_dataloader()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tiktoken.get_encoding("gpt2"),
    )
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
