from dataprocess import SimpleRegexTokenizerV1, SimpleRegexTokenizerV2, get_vocab

vocab = get_vocab()
tokenizerv1 = SimpleRegexTokenizerV1(vocab)
tokenizerv2 = SimpleRegexTokenizerV2(vocab)


def test_tokenizerv1():
    ids = tokenizerv1.encode("How are you?")
    assert tokenizerv1.decode(ids) == "How are you?"


def test_tokenizerv2():
    ids = tokenizerv2.encode("Hello, How are you?")
    assert tokenizerv2.decode(ids) == "<|unk|>, How are you?"


def test_tokenizerv2_endoftext():
    text1 = "Hello, How are you?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join([text1, text2])
    ids = tokenizerv2.encode(text)
    assert (
        tokenizerv2.decode(ids)
        == "<|unk|>, How are you? <|endoftext|> In the sunlit terraces of the <|unk|>."
    )
