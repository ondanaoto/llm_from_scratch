import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor):
        _, seq_len = in_idx.shape
        # shape: [batch, seq_len, emb_dim]
        tok_embeds = self.tok_emb(in_idx)
        # `model.to(device)`をしてもtorch.arangeはdevice上に移動しないので
        # 明示的に指定する必要がある
        # ## なぜ移動しないのか
        # - model.to(device)は__init__内のtorchを対象とする
        # - pos_embに入れる位置idはxによって動的に変更する部分なので__init__には置けない
        # ## __init__内でto(device)によってeviceが変更されるtensorオブジェクトの登録方法
        # self.register_buffer("pos_ids", torch.arange(seq_len))のようにする
        # shape: [seq_len, emb_dim]
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # broadcastされ shape: [batch, seq_len, emb_dim]
        x = tok_embeds + pos_embeds
        # shape変わらず [batch, seq_len, emb_dim]
        x = self.drop_emb(x)
        # shape変わらず [batch, seq_len, emb_dim]
        x = self.trf_blocks(x)
        # shape変わらず [batch, seq_len, emb_dim]
        x = self.final_norm(x)
        # logitsとは，sigmoid関数で確率化される前の実数
        # 取りうる値は実数全体
        # shape: [batch, seq_len, vocab_size]
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm(nn.Module):
    # TODO: broadcastが効くならemb_dimの指定は不要では？
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        # unbiasedフラグに関して
        # 不偏分散ではない計算方法(つまりn-1ではなくnでわる)
        # LLMだとnもn-1もほとんど同じなのでおけ．
        # 採用したのでGPT-2モデルの正規仮装との互換性を維持するためだそう．
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
