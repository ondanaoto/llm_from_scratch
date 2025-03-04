import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import TokenizerInterface


class GPTDatasetV1(Dataset):
    """
    訓練のためのdatasetを作成するクラス
    """

    def __init__(
        self, txt: str, tokenizer: TokenizerInterface, max_length: int, stride: int
    ):
        """訓練のためのdatasetを作成するクラス

        Args:
            txt (str): テキストデータ
            tokenizer (TokenizerInterface): トークナイザ
            max_length (int): 次単語予測のためのヒントの数．
                256などの大きめの値が指定されがち．
            stride (int): トークンのずらし幅．大きいほど訓練データが少なくなる．
                max_lengthと同じにするとバッチ間のオーバーラップを防げる．
                オーバーラップは過適合のリスクを高める
        """
        # 入力変数を格納するリスト
        # idのリストをtensorに変換したものが入る
        self.input_ids = []
        # 目的変数を格納するリスト
        # idのリストをtensorに変換したものが入る
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """訓練データ数を返す

        Returns:
            int: データ数
        """
        return len(self.input_ids)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """訓練データを返す

        Args:
            index (int): データのインデックス

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 入力変数と目的変数のidのtensorで返す．
            入力変数：max_lengthの長さのトークン列のidのtensor．
            目的変数：入力変数が1つずれていて，次の単語が含まれたトークン列のidのtensor
        """
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
    txt: str,
    tokenizer: TokenizerInterface | None = None,
    batch_size: int = 1,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last=True,
    num_workers=0,
):
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
