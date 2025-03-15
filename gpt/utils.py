import torch
import torch.nn as nn


def generate_text_simple(
    gpt_model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int
) -> torch.Tensor:
    """gptモデルを用いて次単語予測を続けさせる

    Args:
        gpt_model (nn.Module): GPTモデル
        idx (torch.Tensor): batch * n_tokensのトークンidのtorch配列
        max_new_tokens (int): 何単語分続けさせるか
        context_size (int): 最大何単語までgpt_modelが取り扱えるのか

    Returns:
        torch.Tensor: 続けた後のトークンid列
    """
    for _ in range(max_new_tokens):
        # 最後context_size分のindexを取得
        # つまり，いくらでもトークン列を生成はできるが，
        # 最初の方の情報を持たないで生成することになる
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            # seq_len = len(idx[0])
            # n_tokens = min(context_size, seq_len)
            # logits.shape = [batch_size, n_tokens, vocab_size]
            logits = gpt_model(idx_cond)

        # logits.shape = [batch_size, vocab_size]
        # 最後の単語のlogitたち
        logits = logits[:, -1, :]

        # dim=-1でsumを取ると1になるように確率化(自然だ)
        # しかし不要な気がする(後でargmaxをとっていて，softmaxは大小関係を変えないので)
        probas = torch.softmax(logits, dim=-1)

        # idx_next.shape = [batch_size, 1] (keepdimしてるので)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # idx.shape = [batch, n_tokens + 1]
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id=None,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            # shape: [batch_size, n_token, vocab_size]
            logits = model(idx_cond)

        # shape: [batch_size, vocab_size]
        # 各バッチの次単語のlogits
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            # k番目に高いlogitsを取得する
            # この値以下のlogitsを-infに変換するために
            min_val = top_logits[:, -1]
            logits = torch.where(
                condition=logits < min_val,
                input=torch.tensor(float("-inf")).to(logits.device),
                other=logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            # 確率分布に従ったidxのサンプリング
            # shape: [batch_size, 1]
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # shape: [batch_size, 1]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # batch全ての最後がeos_idだったらbreak
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
