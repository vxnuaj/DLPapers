<div align = 'center'>
<img src = 'https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-20_at_11.35.53_PM_KroVKVL.png'>
</div>

# DenseNet

Notes and Implementation of DenseNet, proposed on *[Densely Connected Convolutional Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)* by Huang et al.

## Usage

1. Clone the Repo
2. Run `run.py`

    ```python

    import torch
    from torchinfo import summary
    from densenet import DenseNet

    # init randn tensor

    x = torch.randn(size = (3, 3, 224, 224))

    # init model

    model = DenseNet( k = 40, theta = .5)

    # get summary and final output shape

    summary(model, x.size())
    print(f"\nFinal Output Shape: {model(x).size()}")

    ```