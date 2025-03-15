import tiktoken
import torch

from gpt import GPT_CONFIG_124M_SHORT_CONTEXT, GPTModel
from tokenizer.utils import text_to_token_ids


def test_save_load():
    model = GPTModel(GPT_CONFIG_124M_SHORT_CONTEXT)
    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = "Every effot moves you"
    token_ids = text_to_token_ids(text, tokenizer).to(device)
    model.eval()
    logits_before = model(token_ids)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "data/model.pth",
    )
    modela = GPTModel(GPT_CONFIG_124M_SHORT_CONTEXT)
    modela.eval()
    checkpoint = torch.load("data/model.pth", map_location=device)
    modela.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logits_after = modela(token_ids)

    assert torch.allclose(logits_before, logits_after)
