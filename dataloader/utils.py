import tiktoken
from torch.utils.data import DataLoader

from tokenizer import Tokenizer

from .models import GPTDatasetV1


def create_dataloader_v1(
    txt: str,
    tokenizer: Tokenizer | None = None,
    batch_size: int = 1,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last=True,
    num_workers=0,
) -> DataLoader:
    """データローダーの作成

    Args:
        txt (str): 訓練データの作成のためのテキストデータ
        tokenizer (TokenizerInterface, optional): トークナイザ.
            Defaults to tiktoken.get_encoding("gpt-2").
        batch_size (int, optional):
            1回呼び出した時に提出してくれる入力変数と目的変数のサイズ.
            Defaults to 4.
        max_length (int, optional): 入力変数のid列の長さ(Datasetのmax_lengthと同じ).
            Defaults to 256.
        stride (int, optional): データセット作成上のずらし幅．
            大きいほど訓練データは減る．
            (Datasetのstrideと同じ). Defaults to 128.
        shuffle (bool, optional): ランダムに訓練データを提供するか否か.
            Defaults to True.
        drop_last (bool, optional): 指定されたbatch_sizeよりも最後のバッチが短い場合に
            訓練中の損失値のスパイクを防ぐためにそのバッチを除外する. Defaults to True.
        num_workers (int, optional): 前処理に使うCPUプロセスの数. Defaults to 0.

    Returns:
        DataLoader: データローダー．
    """
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader
