# TODO

import torch
from torchinfo import summary
from mobilenetv2 import MobileNetV2

# init model

model = MobileNetV2()

# init randn tensor

x = torch.randn( size = (2, 3, 224, 224))

# run model, get summary, and final output size

summary(model, x.size())
print(f"\nFinal  Output Size: {model(x).size()}")