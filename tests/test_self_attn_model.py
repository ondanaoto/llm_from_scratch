import torch

from attention import SelfAttention_v1, SelfAttention_v2


def test_forwardv1():
    # context_length = 4, d_in = 3
    inputs = torch.rand(4, 3)
    as_v1 = SelfAttention_v1(3, 6)
    assert as_v1(inputs).shape == torch.Size([4, 6])


def test_forwardv2():
    # context_length = 4, d_in = 3
    inputs = torch.rand(4, 3)
    as_v2 = SelfAttention_v2(3, 6)
    assert as_v2(inputs).shape == torch.Size([4, 6])


def test_causal():
    context_length = 4
    d_in = 3
    d_out = 6
    inputs = torch.rand(context_length, d_in)
    as_v2 = SelfAttention_v2(d_in, d_out)
    queries = as_v2.W_query(inputs)
    keys = as_v2.W_key(inputs)

    # content_length * content_length配列
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores, dim=-1)
    assert attn_weights[0].sum() - 1.0 < 0.001
    assert attn_weights[1].sum() - 1.0 > -0.001

    # 右上部分(未来のところ)を0にする
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    # [[1., 0.]
    #  [1., 1.]]
    for i in range(context_length):
        for j in range(context_length):
            if i < j:
                assert mask_simple[i][j] == 0
            else:
                assert mask_simple[i][j] == 1
    masked_simple = attn_weights * mask_simple
    assert masked_simple[0][0] > 0
    assert masked_simple[0][1] == 0

    # 線形に正規化する
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    for i in range(context_length):
        assert masked_simple_norm.sum(dim=-1)[i] - 1.0 > -0.01
        assert masked_simple_norm.sum(dim=-1)[i] - 1.0 < 0.01

    # 本当は，対角線上側を-infにしてsoftmaxすれば必要なタスクは終わる
    # それはCausalAttentionのforwardで実装
    # Dropout maskも追加したい
    # dropoutは訓練時のみ
    # dropoutのタイミングは重みの計算後とvalueベクトルへの適用後
