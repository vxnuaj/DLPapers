import torch
from torchinfo import summary
from inceptionv1 import InceptionV1

# init random tensor
x = torch.randn(1, 3, 224, 224)

# init model

model = InceptionV1()

# model summary
summary(model, input_data = x)
