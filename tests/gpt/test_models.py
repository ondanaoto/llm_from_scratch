import tiktoken
import torch

from dataprocess import TokenizerInterface
from gpt import GPT_CONFIG_124M, DummyGPTModel, LayerNorm

torch.manual_seed(123)


def test_dummygpt():
    tokenizer: TokenizerInterface = tiktoken.get_encoding("gpt2")
    texts = ["Every effort moves you", "Every day holds a"]
    batch = torch.stack([torch.tensor(tokenizer.encode(text)) for text in texts], dim=0)
    assert batch.shape == torch.Size([2, 4])

    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)

    assert logits.shape == torch.Size([*batch.shape, GPT_CONFIG_124M["vocab_size"]])


def test_layernorm():
    batch_size = 2
    emb_dim = 5
    batch_example = torch.rand(batch_size, emb_dim)
    ln = LayerNorm(emb_dim=emb_dim)
    out_ln: torch.Tensor = ln(batch_example)
    assert torch.allclose(out_ln.mean(dim=-1), torch.zeros(batch_size), atol=1e-7)
    # LayerNormでunbiased=Falseにして処理しているのでここもFalseにする
    # 0除算を回避するためにeps=1e-5をvarに足してから
    # 平方根を取って標準偏差を導出している関係で，誤差は少し大きくなる
    assert torch.allclose(
        out_ln.var(dim=-1, unbiased=False), torch.ones(batch_size), atol=1e-3
    )
