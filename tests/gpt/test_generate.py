import tiktoken
import torch

from gpt import GPT_CONFIG_124M, GPTModel
from gpt.utils import generate_text_simple


def test_text_simple():
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)
    start_context = "Hello, I am"
    # n_tokens = len(encoded)
    encoded = tokenizer.encode(start_context)
    # batch_sizeが1のbatchに変換する
    # encoded_tensor = [1, n_tokens]
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # dropoutを無効にする
    model.eval()

    out: torch.Tensor = generate_text_simple(
        gpt_model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    out_idx = out.squeeze().tolist()
    assert out_idx[:4] == encoded
    assert len(out_idx) == 10
    decoded_text = tokenizer.decode(out_idx)
    assert decoded_text.startswith("Hello, I am")
