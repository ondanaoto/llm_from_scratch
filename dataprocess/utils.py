import re


def get_raw_text():
    with open("data/the-verdict.txt") as file:
        text = file.read()
    return text


def print_text_head():
    with open("data/the-verdict.txt") as file:
        text = file.read()

    print(text[:500])


def split_by_simple_regex(text: str) -> list[str]:
    return [token for token in re.split(r'([,.?_!"()\']|--|\s)', text) if token.strip()]


def get_vocab() -> dict[str, int]:
    with open("data/the-verdict.txt") as file:
        text = file.read()

    tokens = split_by_simple_regex(text)
    all_words = sorted(set(tokens))
    vocab = {token: i for i, token in enumerate(all_words)}
    return vocab


def remove_space_after_punctuation(text: str) -> str:
    return re.sub(r"\s+([,.?!()\'])", r"\1", text)
