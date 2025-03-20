import os
import urllib.request
import zipfile
from pathlib import Path


def download_data():
    """
    トークン化を行うテキストファイルを取得するスクリプト．一回だけ呼び出す．
    すると`data`ディレクトリに`the-verdict.txt`がダウンロードされる．
    """

    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )

    file_path = Path("data") / "the-verdict.txt"
    if not file_path.exists():
        os.makedirs("data")
        urllib.request.urlretrieve(url, file_path)


def download_ft_dataset(
    url: str,
    zip_path: Path,
    extracted_path: Path,
    original_file_name: str,
    rename_file_name: str,
):
    data_file_path = extracted_path / rename_file_name
    if data_file_path.exists():
        return

    with urllib.request.urlopen(url) as response:
        with zip_path.open("wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    os.remove(zip_path)

    original_file_path = extracted_path / original_file_name
    os.rename(original_file_path, data_file_path)


if __name__ == "__main__":
    download_data()

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = Path("sms_spam_collection.zip")
    extracted_path = Path("data") / "sms_spam_collection"
    original_file_name = "SMSSpamCollection"
    rename_file_name = "SMSSpamCollection.tsv"

    download_ft_dataset(
        url, zip_path, extracted_path, original_file_name, rename_file_name
    )
