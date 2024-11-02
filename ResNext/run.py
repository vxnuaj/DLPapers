import torch
from torchinfo import summary
from resnext import ResNeXt50

# init randn tensor

x = torch.randn(size = (2, 3, 224, 224))

# init model & get summary

model = ResNeXt50()

summary(model, input_size = x.size())
