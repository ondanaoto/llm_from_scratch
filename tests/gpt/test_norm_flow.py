import torch
import torch.nn as nn


def test_normalize_flow():
    torch.manual_seed(123)
    batch_example = torch.randn([2, 5])
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out: torch.Tensor = layer(batch_example)

    # あとでbroadcast機能で正規化するためにkeepdimする
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    assert mean.shape == torch.Size([2, 1])
    assert var.shape == torch.Size([2, 1])

    # outは正規化(つまり(mean, var) = (0.0, 1.0)になるように線形変換)されていない
    assert not torch.allclose(mean, torch.tensor([0.0, 0.0]), atol=1e-7)
    assert not torch.allclose(var, torch.tensor([1.0, 1.0]), atol=1e-7)

    # meanやvarが out.shape = [2,6]にbroadcastされて各要素で演算される
    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1)
    var = out_norm.var(dim=-1)
    assert torch.allclose(mean, torch.tensor([0.0, 0.0]), atol=1e-7), (
        "mean is not close enough to zero"
    )
    assert torch.allclose(var, torch.tensor([1.0, 1.0]), atol=1e-7), (
        "var is not close enough to one"
    )
