import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator


def plot_losses(
    epochs_seen: torch.Tensor,
    tokens_seen: list[int],
    train_losses: list[float],
    val_losses: list[float],
) -> None:
    # 幅5インチ, 高さ3インチのサイズで全体の図を作成
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ax1とy軸を共有する新たなx軸を作成する
    ax2 = ax1.twiny()
    # 描画するものの，透明度0にすることで線は表示されない
    # 上部のx軸のスケールを設定するためのトリック．
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    # 各要素が重ならないように自動的にレイアウトを整えてくれる
    fig.tight_layout()
    # 描画
    plt.show()
