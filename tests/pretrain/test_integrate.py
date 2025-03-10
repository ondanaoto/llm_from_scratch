import tiktoken
import torch

from gpt import GPT_CONFIG_124M_SHORT_CONTEXT, GPTModel
from gpt.utils import generate_text_simple
from tokenizer.utils import text_to_token_ids, token_ids_to_text


def test_integrate():
    torch.manual_seed(42)
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M_SHORT_CONTEXT)
    token_ids = generate_text_simple(
        gpt_model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M_SHORT_CONTEXT["context_length"],
    )
    expected = (
        "Every effort moves youodonicle ' directly inflamm honeyopoly Kw ply benefit"
    )
    assert token_ids_to_text(token_ids, tokenizer) == expected
