from dataloader.utils import create_dataloader_v1, create_spam_dataloaders
from utils import get_raw_text


def test_dataloader():
    dataloader = create_dataloader_v1(
        get_raw_text(),
        batch_size=4,
        max_length=10,
        shuffle=False,
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    # 入力変数のバッチと目的変数のバッチが返されるので長さは2
    assert len(first_batch) == 2
    # バッチサイズが4で，max_lengthが10なので，(4, 10)のshapeが返される
    assert first_batch[0].shape == (4, 10)
    assert first_batch[1].shape == (4, 10)


def test_spam_dataset():
    import torch
    train_loader, _, _ = create_spam_dataloaders()

    for input_batch, target_batch in train_loader:
        pass

    assert input_batch.shape == torch.Size([8,120])
    assert target_batch.shape == torch.Size([8])
