import torch

from gpt import (
    CausalAttention,
    MultiHeadAttention,
    SelfAttention_v1,
    SelfAttention_v2,
)


def test_forwardv1():
    # context_length = 4, d_in = 3
    inputs = torch.rand(4, 3)
    as_v1 = SelfAttention_v1(3, 6)
    assert as_v1(inputs).shape == torch.Size([4, 6])


def test_forwardv2():
    # batch_size = 16, num_tokens = 4, d_in = 3
    inputs = torch.rand(16, 4, 3)
    as_v2 = SelfAttention_v2(3, 6)
    context_vec = as_v2(inputs)
    assert context_vec.shape == torch.Size([16, 4, 6])


def test_causal():
    # batch_size = 16, num_tokens = 4, d_in = 3
    inputs = torch.rand(16, 4, 3)
    causal = CausalAttention(d_in=3, d_out=6, context_length=32, dropout=0.5)
    context_vec = causal(inputs)
    assert context_vec.shape == torch.Size([16, 4, 6])


def test_multi():
    # batch_size: 4
    # num_token: 6
    # d_in: 4
    inputs = torch.rand(4, 6, 4)
    attn = MultiHeadAttention(
        d_in=4, d_out=8, context_length=768, num_heads=4, dropout=0.5
    )
    context_vec = attn(inputs)
    assert context_vec.shape == torch.Size([4, 6, 8])
