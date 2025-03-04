from abc import ABC, abstractmethod

from .utils import remove_space_after_punctuation, split_by_simple_regex


class TokenizerInterface(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError


class SimpleRegexTokenizerV1(TokenizerInterface):
    """
    与えられた語彙の対応を用いてエンコード・デコードを行うトークナイザー．
    単語への分割は正規表現を用いたルールベース．
    未知の語彙には対応できない．
    """

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: token for token, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        tokens = split_by_simple_regex(text)
        ids = [self.str_to_int[token] for token in tokens]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        text = remove_space_after_punctuation(text)
        return text


class SimpleRegexTokenizerV2(TokenizerInterface):
    """
    `SimpleRegexTokenizerV1`と基本は同じだが，語彙の外の単語には`<|unk|>`を割り当てる．
    また，無関係な文の繋がりに挟む`<|endoftext|>`を語彙に追加してある．
    """

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        max_id = len(vocab)
        self.str_to_int["<|unk|>"] = max_id
        self.str_to_int["<|endoftext|>"] = max_id + 1
        self.int_to_str = {i: token for token, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        split_result = split_by_simple_regex(text)
        ids = [
            self.str_to_int.get(token, self.str_to_int["<|unk|>"])
            for token in split_result
        ]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        text = remove_space_after_punctuation(text)
        return text
