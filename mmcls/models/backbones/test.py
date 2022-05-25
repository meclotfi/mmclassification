import torch
from mobilevit import MobileViTBlock,MobileViT
import torch
print(torch.__version__)

model=MobileViTBlock(in_channels=32,num_heads=4,ffn_dim=64)
t=torch.rand(size=(1,32,28,28))
print(model.forward(t).shape)

"""
model=MobileViT()
t=torch.rand(size=(1,3,256,256))
outs=model.forward(t)
print([out.shape for out in outs])
"""