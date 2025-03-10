import tiktoken
import torch

from gpt import GPT_CONFIG_124M_SHORT_CONTEXT, GPTModel
from tokenizer.utils import text_to_token_ids


def test_compute_loss():
    torch.manual_seed(42)
    model = GPTModel(GPT_CONFIG_124M_SHORT_CONTEXT)
    texts = ["every effort moves", "I really like"]
    target_texts = [" effort moves you", " really like chocolate"]
    tokenizer = tiktoken.get_encoding("gpt2")
    inputs = torch.cat([text_to_token_ids(text, tokenizer) for text in texts], dim=0)
    targets = torch.cat(
        [text_to_token_ids(text, tokenizer) for text in target_texts], dim=0
    )

    with torch.no_grad():
        # shape: [batch, seq_len, vocab_size]
        # = [2, 3, 50257]
        logits: torch.Tensor = model(inputs)

    # shape: [batch*seq_len, vocab_size] = [6, 50257]
    logits_flat = logits.flatten(0, 1)
    # shape: [batch*seq_len] = [6]
    target_flat = targets.flatten()
    loss_manual = compute_loss_manual(logits_flat, target_flat)
    loss = torch.nn.functional.cross_entropy(logits_flat, target_flat)
    assert torch.allclose(loss, loss_manual, atol=1e-4)


def compute_loss_manual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probas = torch.softmax(logits, dim=-1)
    n_tokens = len(logits)
    # 参考：advanced_indexing
    # https://chatgpt.com/share/67ce672d-40d4-800b-a06a-675292447e2a
    target_probas = probas[list(range(n_tokens)), targets]

    loss = -torch.log(target_probas).mean()
    return loss
