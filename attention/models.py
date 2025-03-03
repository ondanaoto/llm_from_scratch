import torch
import torch.nn as nn


# バッチ対応してない...
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x: torch.Tensor):
        """forward計算

        Args:
            x (torch.Tensor):
            (content_length: コンテクスト長) * (d_in: 単語埋め込みベクトル次元)の配列が
            期待される
        Returns:
            context_vec (torch.Tensor): (context_length * d_out)の配列
        """
        # それぞれ (context_length * d_out)
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # context_length * context_lengthの配列
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # context_length * d_outの配列
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor):
        """forward計算

        Args:
            x (torch.Tensor):
            (batch_size) * (content_length: コンテクスト長) * (d_in: 単語埋め込み
            ベクトル次元)の配列が
            期待される
        Returns:
            context_vec (torch.Tensor): (num_tokens * d_out)の配列
        """
        # それぞれ (num_tokens * d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # num_tokens * num_tokensの配列
        attn_scores = queries @ keys.transpose(1, 2)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # num_tokens * d_outの配列
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    def __init__(
        self, d_in: int, d_out: int, context_length: int, dropout: float, qkv_bias=False
    ):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # dropoutの割合を引数にもつ
        # self.dropoutはランダムにパラメータを0にし, 
        # 成分の和が一致するようにスケーリングする．
        # 例えばdropout=0.5の場合, 0.5の確率でパラメータが0になり，
        # 残りのパラメータは2倍される．
        self.dropout = nn.Dropout(dropout)
        # register_buffer: 最適化しないパラメータを指定する
        # [[0, 1, 1],
        #  [0, 0, 1],
        #  [0, 0, 0]]
        # のようなマスクを作成
        # self.maskでアクセスできる
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
                ),
        )

    def forward(self, x):
        """forward

        Args:
            x (torch.Tensor): batch_size * content_length * d_in 配列
        """
        b, num_tokens, d_in = x.shape
        # batch_size次元部分はブロードキャストされる
        # よってkeysたちはbatch_size * num_tokens * d_out配列
        keys = self.W_key(x)
        queris = self.W_query(x)
        values = self.W_value(x)

        # queries: batch_size * num_tokens * d_out
        # keys: batch_size * num_tokens * d_out
        # keys.transpose(1, 2): batch_size * d_out * num_tokens
        # -> attn_scores: batch_size * num_tokens * num_tokens
        # https://discuss.pytorch.org/t/how-does-the-sign-work-in-this-instance/11232
        # last two dimensionsに対してmatmulが行われる
        # 前半全体はbatchとして認識されbroadcastされる
        attn_scores = queris @ keys.transpose(1, 2)

        # maskがtrueの部分を-infにする
        # masked_fill_の第一引数は, attn_scoresにbroadcastableでなければならない
        # 実際にmaskはbatchにbroadcastされる
        attn_scores.masked_fill_(
            # self.maskはcontent_lengthサイズなので，実際のxのサイズに合わせる．
            self.mask.bool()[:num_tokens, :num_tokens],
            float("-inf"),
        )
        # 現在のattn_scoresの様子：
        # shape: batch_size * num_tokens * num_tokens
        # 各batchの中身において，上三角部分が-infになっている

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
