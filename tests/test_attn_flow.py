import torch

context_length = 16
d_in = 4


def test_simple_attention():
    token_pos_vec = torch.rand(context_length, d_in)

    # `@`は行列演算．c*o行列とo*c行列の積はc*cになる．
    d_out = d_in
    query_vec = token_pos_vec
    key_vec = token_pos_vec
    var_mat = query_vec @ key_vec.T
    assert var_mat.shape == torch.Size([context_length, context_length])

    # softmaxはn次元配列からn次元配列の変換を行う．
    # 何次元目に沿った和が1になるようにするかを指定する．
    # dim=-1としたので，dim=-1の要素の総和が1になる．
    # [[0.2, 0.8],
    #  [0.3, 0.7]]みたいな
    atten_weight = torch.softmax(var_mat, dim=-1)
    assert atten_weight.sum(dim=-1)[0] - 1.0 > -0.001
    assert atten_weight.sum(dim=-1)[0] - 1.0 < 0.001

    value_vec = token_pos_vec
    context_vecs = atten_weight @ value_vec
    assert context_vecs.shape == torch.Size([context_length, d_out])


def test_trainable_attention():
    d_out = 3
    token_pos_vec = torch.rand(context_length, d_in)

    W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
    query_vec = token_pos_vec @ W_query
    key_vec = token_pos_vec @ W_key
    var_mat = query_vec @ key_vec.T
    assert var_mat.shape == torch.Size([context_length, context_length])

    # 単語埋め込みベクトル次元の平方根で割って正規化してsoftmaxする
    assert d_out == key_vec.shape[-1]
    atten_weight = torch.softmax(var_mat / key_vec.shape[-1] ** 0.5, dim=-1)
    assert atten_weight.sum(dim=-1)[0] - 1.0 > -0.001
    assert atten_weight.sum(dim=-1)[0] - 1.0 < 0.001

    W_value = torch.nn.Parameter(torch.rand(d_in, d_out))
    value_vec = token_pos_vec @ W_value
    context_vecs = atten_weight @ value_vec
    assert context_vecs.shape == torch.Size([context_length, d_out])
