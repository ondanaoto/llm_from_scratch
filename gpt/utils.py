import torch
import torch.nn as nn


def generate_text_simple(
    gpt_model: nn.Module, idx: list[list[int]], max_new_tokens: int, context_size: int
) -> list[int]:
    for _ in range(max_new_tokens):
        # 最後context_size分のindexを取得
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
        probas = torch.softmax(logits, dim=-1)

        # idx_next.shape = [batch_size, 1] (keepdimしてるので)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # idx.shape = [batch, n_tokens + 1]
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
