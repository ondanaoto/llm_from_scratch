import torch

from gpt import GPT_CONFIG_124M_SHORT_CONTEXT, GPTModel

model = GPTModel(GPT_CONFIG_124M_SHORT_CONTEXT)
num_classes = 2

for param in model.parameters():
    param.requires_grad = False

model.out_head = torch.nn.Linear(
    in_features=GPT_CONFIG_124M_SHORT_CONTEXT["emb_dim"], out_features=num_classes
)

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True
