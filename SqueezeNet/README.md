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
    from torchinfo import summary
    from squeezenet import SqueezeNet

    # init randn tensor

    x = torch.randn( size = (2, 3, 224, 224))

    # init model

    model = SqueezeNet(

        base_e = 128, 
        incr_e = 128,
        pct_3x3 = .5,
        freq = 2,
        sr = .125

    )

    # run model, get summary, get final output shape

    summary(model, x.size())
    print(f"\nFinal Model Size: {model(x).size()}")



    ```