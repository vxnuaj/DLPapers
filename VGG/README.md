## VGGNet

Notes and implementation of the VGGNet, proposed on *"Very Deep Convolutional Networks for Large-Scale Image Recognition
"* by Simonyan & Zisserman.

## Usage

1. Clone the Repo
2. Install pytorch and torchinfo
   ```zsh
    pip install torch torchvision
    pip install torchinfo
   ```
3. Run `run_vggX.py` variants

    `run_vgg11.py`

    ```python

    import torch
    from torchinfo import summary
    from vgg import VGG11

    # init random shape
    x = torch.randn(1, 3, 224, 224)

    # init model and run a forward pass.
    model = VGG11()
    y = model.forward(x)

    summary(model, input_size = x.size()) # correct!

    ```