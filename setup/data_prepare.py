def download_data():
    """
    トークン化を行うテキストファイルを取得するスクリプト．一回だけ呼び出す．
    すると`data`ディレクトリに`the-verdict.txt`がダウンロードされる．
    """

    import os
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )

    if not os.path.exists("data"):
        os.makedirs("data")
    file_path = "data/the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)


if __name__ == "__main__":
    download_data()
