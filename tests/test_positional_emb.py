import torch

from dataprocess import create_dataloader_v1, get_raw_text

def test_positional_emb():
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    max_length = 4
    batch_size = 8
    dataloader = create_dataloader_v1(
        get_raw_text(),
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )

    data_iter = iter(dataloader)
    inputs, _ = next(data_iter)
    assert inputs.shape == torch.Size([batch_size, max_length])

    token_embeddings = token_embedding_layer(inputs)
    assert token_embeddings.shape == torch.Size([batch_size, max_length, output_dim])

    # 入力されるトークン数
    context_length = max_length
    # コンテクスト長の分だけ，埋め込み次元ベクトルを作成して最適化したいので絶対位置埋め込みベクトルの層を作成．
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    # 層の重みを取得
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    assert pos_embeddings.shape == torch.Size([context_length, output_dim])

    # pos_embeddingsは各バッチに対してブロードキャストされて足される
    # ブロードキャストとは，両者を同じ形状にする操作をして演算を各要素で行う機能．
    # ブロードキャスト則：https://pytorch.org/docs/stable/notes/broadcasting.html
    # # 1. どちらも1次元以上である
    # # 2. 後続から比較して, k番目までブロードキャスタブルだったとする．
    # #    k-1番目において「両者が一致」「一方が1」「一方が存在しない」ならk-1番目までブロードキャスタブル
    # # 3. 2による再帰的な確認の結果全てOKならブロードキャスタブル．
    # 今回の場合：ブロードキャスタブル！
    # # どちらも1次元以上のtorch.tensorである
    # # お尻2つ(max_length, output_dim)は一致している．先頭を比較するとpos_embeddingの先頭がない状態なので2の条件を全て満たす
    input_embeddings = token_embeddings + pos_embeddings
    assert input_embeddings.shape == torch.Size([batch_size, max_length, output_dim])
