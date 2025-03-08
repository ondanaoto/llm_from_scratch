import torch
import torch.nn as nn

from gpt import ExampleShortCutDeepNN


def print_gradients(model: nn.Module, x: torch.Tensor, target: torch.Tensor):
    """loss計算時のモデルのパラメータの勾配をprintする"""
    output = model(x)

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


def test_without_shortcut_vanish_loss():
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1.0, 0.0, -1.0]])

    model_without_shortcut = ExampleShortCutDeepNN(
        layer_sizes=layer_sizes, use_shortcut=False
    )

    target = torch.tensor([[0.0]])

    print("\nWithout shortcuts, we see that the gradients are gradually decreasing")
    print_gradients(model_without_shortcut, sample_input, target)


def test_with_shortcut_retains_loss():
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1.0, 0.0, -1.0]])

    model_with_shortcut = ExampleShortCutDeepNN(
        layer_sizes=layer_sizes, use_shortcut=True
    )

    target = torch.tensor([[0.0]])

    print("\nWith shortcuts, we see that the gradients are stable")
    print_gradients(model_with_shortcut, sample_input, target)
