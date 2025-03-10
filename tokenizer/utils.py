import torch

from .models import TokenizerInterface


def text_to_token_ids(text: str, tokenizer: TokenizerInterface) -> torch.Tensor:
    encoded: list[int] = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # shape: (1, seq_len)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: TokenizerInterface) -> str:
    """token_idsをデコードしてテキストに変換する
    shapeは(1, seq_len)であることを想定している

    Returns:
        _type_: _description_
    """
    token_ids = token_ids.squeeze(0)
    text = tokenizer.decode(token_ids.tolist())
    return text
