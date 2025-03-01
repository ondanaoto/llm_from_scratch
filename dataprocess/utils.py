def get_raw_text():
    with open("data/the-verdict.txt") as file:
        text = file.read()
    return text
