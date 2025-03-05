import torch

num_tokens = 16
d_in = 4


def test_simple_attention():
    token_pos_vec = torch.rand(num_tokens, d_in)

    # `@`は行列演算．c*o行列とo*c行列の積はc*cになる．
    d_out = d_in
    query_vec = token_pos_vec
    key_vec = token_pos_vec
    var_mat = query_vec @ key_vec.T
    assert var_mat.shape == torch.Size([num_tokens, num_tokens])

    # softmaxはn次元配列からn次元配列の変換を行う．
    # 何次元目に沿った和が1になるようにするかを指定する．
    # dim=-1としたので，dim=-1の要素の総和が1になる．
    # [[0.2, 0.8],
    #  [0.3, 0.7]]みたいな
    atten_weight = torch.softmax(var_mat, dim=-1)
    assert torch.allclose(
        atten_weight.sum(dim=-1),
        torch.ones(
            num_tokens,
        ),
        atol=1e-7,
    )

    value_vec = token_pos_vec
    context_vecs = atten_weight @ value_vec
    assert context_vecs.shape == torch.Size([num_tokens, d_out])


def test_trainable_attention():
    d_out = 3
    token_pos_vec = torch.rand(num_tokens, d_in)

    W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
    query_vec = token_pos_vec @ W_query
    key_vec = token_pos_vec @ W_key
    var_mat = query_vec @ key_vec.T
    assert var_mat.shape == torch.Size([num_tokens, num_tokens])

    # 単語埋め込みベクトル次元の平方根で割って正規化してsoftmaxする
    assert d_out == key_vec.shape[-1]
    atten_weight = torch.softmax(var_mat / key_vec.shape[-1] ** 0.5, dim=-1)
    assert torch.allclose(
        atten_weight.sum(dim=-1),
        torch.ones(
            num_tokens,
        ),
        atol=1e-7,
    )

    W_value = torch.nn.Parameter(torch.rand(d_in, d_out))
    value_vec = token_pos_vec @ W_value
    context_vecs = atten_weight @ value_vec
    assert context_vecs.shape == torch.Size([num_tokens, d_out])


def test_causal():
    from attention import SelfAttention_v2

    batch_size = 16
    num_tokens = 4
    d_in = 3
    d_out = 6
    inputs = torch.rand(batch_size, num_tokens, d_in)
    as_v2 = SelfAttention_v2(d_in, d_out)
    queries = as_v2.W_query(inputs)
    keys = as_v2.W_key(inputs)

    # batch_size * content_length * content_length配列
    attn_scores = queries @ keys.transpose(1, 2)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    assert torch.allclose(
        attn_weights.sum(dim=-1),
        torch.ones(
            num_tokens,
        ),
        atol=1e-7,
    )

    # 引数の行列右上部分(未来のところ)を0にして下三角行列を作った
    mask_simple = torch.tril(torch.ones(batch_size, num_tokens, num_tokens))
    # [[1., 0.]
    #  [1., 1.]]
    for b in range(batch_size):
        for i in range(num_tokens):
            for j in range(num_tokens):
                if i < j:
                    assert mask_simple[b][i][j] == 0
                else:
                    assert mask_simple[b][i][j] == 1
    masked_simple = attn_weights * mask_simple
    assert masked_simple[3][0][0] > 0
    assert masked_simple[3][0][1] == 0

    # 線形に正規化する
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    assert torch.allclose(
        masked_simple_norm.sum(dim=-1), torch.ones(batch_size, num_tokens), atol=1e-7
    )

    # 本当は，対角線上側を-infにしてsoftmaxすれば必要なタスクは終わる
    # それはCausalAttentionのforwardで実装
    # Dropout maskも追加したい
    # dropoutは訓練時のみ
    # dropoutのタイミングは重みの計算後とvalueベクトルへの適用後
