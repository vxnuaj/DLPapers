<div align = 'center'>
<img src = 'https://miro.medium.com/v2/resize:fit:2000/1*UItPkoIvPZR5iXgzVgap6g.png'>
</div>

## AlexNet


Implementation of the original AlexNet, propsoed on *[ImageNet Classification with Deep Convolutional
Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)* by Krizhevsky et al.

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