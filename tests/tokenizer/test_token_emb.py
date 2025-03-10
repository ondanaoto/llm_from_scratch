import torch

from dataloader.utils import create_dataloader_v1
from tokenizer import (
    SimpleRegexTokenizerV2,
)
from utils import get_raw_text, get_vocab


def test_embedding():
    vocab = get_vocab()
    tokenizer = SimpleRegexTokenizerV2(vocab)
    vocab_size = len(vocab)
    output_dim = 16
    max_length = 256
    batch_size = 4

    # tokenizerのvocab_sizeだけoutput_dim次元ベクトルを作成する
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    # tokenizerがgpt2のままだと，vocab_sizeを超えたidが割り振られてしまうことがあり，
    # embedding_layerの入力の範囲を超えてしまう．
    dataloader = create_dataloader_v1(
        get_raw_text(), tokenizer, max_length=max_length, batch_size=batch_size
    )
    data_iter = iter(dataloader)
    input_ids, _ = next(data_iter)
    assert input_ids.shape == torch.Size([batch_size, max_length])
    assert embedding_layer(input_ids).shape == torch.Size(
        [batch_size, max_length, output_dim]
    )
