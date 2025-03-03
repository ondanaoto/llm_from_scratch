import torch

from attention import SelfAttention_v1, SelfAttention_v2, CausalAttention


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