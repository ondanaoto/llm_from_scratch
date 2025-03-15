import matplotlib.pyplot as plt
import pytest
import tiktoken
import torch

from gpt import GPT_CONFIG_124M_SHORT_CONTEXT, GPTModel
from gpt.utils import generate
from tokenizer.utils import text_to_token_ids, token_ids_to_text
from utils import softmax_with_temperature

vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)


def test_random_decode():
    torch.manual_seed(123)
    probas = torch.softmax(next_token_logits, dim=0)
    sample = [torch.multinomial(probas, num_samples=1).item() for _ in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    expected_freq_dict = {
        "closer": 73,
        "every": 0,
        "effort": 0,
        "forward": 582,
        "inches": 2,
        "moves": 0,
        "pizza": 0,
        "toward": 343,
    }
    for i, freq in enumerate(sampled_ids):
        assert freq == expected_freq_dict[inverse_vocab[i]]


@pytest.mark.view
def test_temperature():
    temperatures = [1, 0.1, 5]
    scaled_probas = [
        softmax_with_temperature(next_token_logits, T) for T in temperatures
    ]
    # 0, 1, 2, ..., 8
    x = torch.arange(len(vocab))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        _ = ax.bar(
            x + i * bar_width, scaled_probas[i], bar_width, label=f"Temperature = {T}"
        )
    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.show()


def test_topk_sampling():
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    assert torch.allclose(top_logits, torch.tensor([6.7500, 6.2800, 4.5100]))
    assert top_pos[0] == 3
    assert top_pos[1] == 7
    assert top_pos[2] == 0

    new_logits = torch.where(
        condition=next_token_logits < top_logits[-1],
        input=torch.tensor(float("-inf")),
        other=next_token_logits,
    )
    assert torch.allclose(
        new_logits,
        torch.tensor(
            [
                4.5100,
                float("-inf"),
                float("-inf"),
                6.7500,
                float("-inf"),
                float("-inf"),
                float("-inf"),
                6.2800,
                float("-inf"),
            ]
        ),
    )
    topk_probas = torch.softmax(new_logits, dim=0)
    assert torch.allclose(
        topk_probas,
        torch.tensor(
            [
                0.0615,
                0.0000,
                0.0000,
                0.5775,
                0.0000,
                0.0000,
                0.0000,
                0.3610,
                0.0000,
            ]
        ),
        atol=1e-4,
    )


def test_generate():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M_SHORT_CONTEXT)
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M_SHORT_CONTEXT["context_length"],
        top_k=25,
        temperature=1.4,
    )
    expected = (
        "Every effort moves you Samoa Cubsdebug Saga reimb "
        "ShannonBow eight disable elemental Added scrapsassium Newsashes"
    )
    assert token_ids_to_text(token_ids, tokenizer) == expected
