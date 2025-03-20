import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer


class GPTDatasetV1(Dataset):
    """
    訓練のためのdatasetを作成するクラス
    """

    def __init__(self, txt: str, tokenizer: Tokenizer, max_length: int, stride: int):
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


class SpamDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        tokenizer: Tokenizer,
        max_length=None,
        pad_token_id: int = 50256,
    ):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = max(
                *[len(encoded_text) for encoded_text in self.encoded_texts]
            )
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = int(self.data.iloc[index]["Label"])
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)
