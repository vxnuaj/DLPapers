<div align = 'center'>
<img src = 'https://miro.medium.com/v2/resize:fit:930/1*ONk0HfLLjDcUhUjuu8iq1w.png'>
</div>

## SqueezeNet

Notes and PyTorch Implementation of SqueezeNet, proposed on *[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360)*

### Index

1. [Paper Notes](squeezenet.md)
2. [Implementation](squeezenet.py)

## Usage

1. Clone the Repo
2. Run `run.py`

    ```python

    import torch
    from alexnet import AlexNet

    # init random shape
    x = torch.randn(1, 3, 224, 224)

    # init model and run a forward pass.
    model = AlexNet()
    y = model.forward(x)

    print(f"AlexNet Output Shape: {y.size()}")


    ```