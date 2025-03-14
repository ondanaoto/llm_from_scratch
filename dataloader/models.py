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
