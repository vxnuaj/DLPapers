<div align = 'center'>
<img src = 'https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-022-19639-x/MediaObjects/41598_2022_19639_Fig3_HTML.png'/>
</div>

# Xception: Deep Learning with Depthwise Separable Convolutions

Notes and Implementation of Xception, proposed on *"Xception: Deep Learning with Depthwise Seperable Convolutions"* by Francois Chollet

### Index

1. [Notes](xception.md)
2. [Implementation](xception.py)

## Usage

1. Clone the repo
2. Run `run.py`

    ```python

    import torch
    from torchinfo import summary
    from xception import Xception

    # dummy randn tensor

    x = torch.randn(size = (2, 3, 299, 299))

    # init model

    model = Xception( fc = True ) # including optional fc layers

    # get model summary and final output shape

    summary(model, x.size())
    print(f"\nFinal Output Size: {model(x).size()}")



    ```


