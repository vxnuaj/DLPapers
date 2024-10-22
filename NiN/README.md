<div align = 'center'>
<img src = 'https://gojay.top/gallery/thumbnails/NIN.png'>
</div>

## Network in Network

Notes and implementation of the NiN blocks, proposed on *"Network in Network"* by Lin et al.

## Usage

1. Clone the Repo
2. Install pytorch and torchinfo
   ```zsh
    pip install torch torchvision
    pip install torchinfo
   ```
3. Run `run.py` variants

    `run_vgg11.py`

    ```python

    import torch
    from torchinfo import summary
    from nin import NiN

    # init model
    model = NiN()

    # init random tensor
    x = torch.randn(size = (1, 3, 224, 224))

    summary(model, input_size = x.size()) 

    ```