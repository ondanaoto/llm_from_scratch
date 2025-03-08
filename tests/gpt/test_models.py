import tiktoken
import torch

from dataprocess import TokenizerInterface
from gpt import GPT_CONFIG_124M, DummyGPTModel, GPTModel, LayerNorm, TransformerBlock


def test_dummygpt():
    tokenizer: TokenizerInterface = tiktoken.get_encoding("gpt2")
    texts = ["Every effort moves you", "Every day holds a"]
    batch = torch.stack([torch.tensor(tokenizer.encode(text)) for text in texts], dim=0)
    assert batch.shape == torch.Size([2, 4])

    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)

    assert logits.shape == torch.Size([*batch.shape, GPT_CONFIG_124M["vocab_size"]])


def test_layernorm():
    torch.manual_seed(123)
    batch_size = 2
    emb_dim = 5
    batch_example = torch.rand(batch_size, emb_dim)
    ln = LayerNorm()
    out_ln: torch.Tensor = ln(batch_example)
    assert torch.allclose(out_ln.mean(dim=-1), torch.zeros(batch_size), atol=1e-5)
    # LayerNormでunbiased=Falseにして処理しているのでここもFalseにする
    # 0除算を回避するためにeps=1e-5をvarに足してから
    # 平方根を取って標準偏差を導出している関係で，誤差は少し大きくなる
    assert torch.allclose(
        out_ln.var(dim=-1, unbiased=False), torch.ones(batch_size), atol=1e-3
    )


def test_transformer():
    x = torch.rand(2, 4, GPT_CONFIG_124M["emb_dim"])
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)

    assert x.shape == torch.Size([2, 4, 768])
    assert output.shape == torch.Size([2, 4, 768])


gpt_model = GPTModel(GPT_CONFIG_124M)


def test_gpt():
    batch = torch.tensor([[6109, 3626, 6100, 3451], [6109, 1110, 6622, 257]])

    out = gpt_model(batch)
    assert batch.shape == torch.Size([2, 4])
    assert out.shape == torch.Size([2, 4, GPT_CONFIG_124M["vocab_size"]])


def test_gpt_numel():
    total_params = sum(p.numel() for p in gpt_model.parameters())
    assert total_params == 162_971_186

    # nn.Embeddingのshapeがtransposeじゃないのも，扱いやすさが理由にありそう．
    # nn.Embeddingを利用するのはindexを指定して埋め込みベクトルを取得することなので，
    # 第一成分はindex, 第二成分にweightが入るという順番が適切．
    assert gpt_model.tok_emb.weight.shape == torch.Size(
        [GPT_CONFIG_124M["vocab_size"], GPT_CONFIG_124M["emb_dim"]]
    )
    # nn.Linearは, transposeを右から作用させるので左作用．
    # だけどnn.Linearで指定する引数の順番がtransposeした後の順番だから
    # weightのshapeと引数指定の順番が逆になっている．
    # nn.Linearで指定する引数の順番がtransposeした後の順番のメリットとしては
    # 作用以前の次元が先になるので直感的になる点が挙げられる．
    assert gpt_model.out_head.weight.shape == torch.Size(
        [GPT_CONFIG_124M["vocab_size"], GPT_CONFIG_124M["emb_dim"]]
    )


def test_total_params_gpt2():
    total_params_gpt2 = sum(p.numel() for p in gpt_model.parameters())
    assert total_params_gpt2 == 162_971_186

    # 一つのパラメータがfloat32と仮定すると，1byte = 8bitより
    total_size_bytes = total_params_gpt2 * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    # weight保存に必要なメモリは621.69MB
    assert round(total_size_mb, 2) == 621.69
