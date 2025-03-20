from pathlib import Path

import pandas as pd


def create_balanced_dataset(df: pd.DataFrame):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    return balanced_df


def random_split(
    df: pd.DataFrame, train_frac: float, validation_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """訓練・検証・テスト用のデータフレームの作成"""
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def main():
    train_path = Path("data") / "train.csv"
    validation_path = Path("data") / "validation.csv"
    test_path = Path("data") / "test.csv"
    if train_path.exists() and validation_path.exists() and test_path.exists():
        return

    data_file_path = Path("data") / "sms_spam_collection" / "SMSSpamCollection.tsv"
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

    balanced_df = create_balanced_dataset(df)

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    train_df.to_csv(train_path, index=None)
    validation_df.to_csv(validation_path, index=None)
    test_df.to_csv(test_path, index=None)


if __name__ == "__main__":
    main()
