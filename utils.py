import re

import torch


def split_by_simple_regex(text: str) -> list[str]:
    return [token for token in re.split(r'([,.?_!"()\']|--|\s)', text) if token.strip()]


def get_raw_text():
    with open("data/the-verdict.txt") as file:
        text = file.read()
    return text


def get_vocab() -> dict[str, int]:
    with open("data/the-verdict.txt") as file:
        text = file.read()

    tokens = split_by_simple_regex(text)
    all_words = sorted(set(tokens))
    vocab = {token: i for i, token in enumerate(all_words)}
    return vocab


def softmax_with_temperature(logits: torch.Tensor, temperature: float):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
