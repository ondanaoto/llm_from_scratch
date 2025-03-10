import tiktoken


def test_tiktoken():
    tokenizer = tiktoken.get_encoding("gpt2")
    text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        " of someunknownPlace."
    )
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    assert tokenizer.decode([integers[0]]) == "Hello"
    assert tokenizer.decode([integers[1]]) == ","
    assert tokenizer.decode([integers[2]]) == " do"
    assert tokenizer.decode([integers[3]]) == " you"
    assert tokenizer.decode([integers[4]]) == " like"
    assert tokenizer.decode([integers[5]]) == " tea"
    assert tokenizer.decode([integers[6]]) == "?"
    assert tokenizer.decode([integers[7]]) == " "
    assert tokenizer.decode([integers[8]]) == "<|endoftext|>"
    # BPEトークナイザの合計サイズが50257で，<|endoftoken|>は最後に割り当てられている．
    assert integers[8] == 50256
    # `someunknownPlace`という未知の単語が含まれていても元に戻る．
    # これは，BPEトークナイザが，語彙に含まれていない単語をより小さなサブワード単位か，
    # バラバラの文字に分解して対応するから．
    assert tokenizer.decode(integers) == text


def test_tiktoken_exercise():
    hoge = "Akwirw ier"
    tokenizer = tiktoken.get_encoding("gpt2")
    integers = tokenizer.encode(hoge)
    assert integers == [33901, 86, 343, 86, 220, 959]
    assert tokenizer.decode(integers) == hoge
