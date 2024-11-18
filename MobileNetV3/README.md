<img src = 'https://academic.hep.com.cn//article/2023/2095-2430/2095-2430-17-5-732/fsc-22965-zz-fig2.jpg'/>

# Searching for MobileNetV3

Notes and PyTorch Implementation of MobileNetV3, proposed by Howard et al., on *["Searching for MobileNetV3"](https://arxiv.org/abs/1905.02244)*

### Index

1. [Notes](notes.md)
2. [Implementation](mobilenetv3.py)

### Usage

1. Clone the Repo
2. Run `run.py`

    ```python
    import torch
    from torchinfo import summary
    from mobilenetv3 import MobileNetV3Large

    # init randn tensor

    x = torch.randn(size = (2, 3, 224, 224))

    # init model

    model = MobileNetV3Large( k = 1000 ) # for 1k imagenet classes

    # get model summary and final output shape

    summary(model, x.size())
    print(f"\nFinal Output Shape: {model(x).size()}")
    ```