<div align = 'center'>
<img src = 'https://miro.medium.com/v2/resize:fit:1400/1*QK1TVTasgdRYpVC31CuPyA.png'>
</div>

# SENet

Notes and PyTorch Implementation of "Squeeze-and-Excitation Networks" by Hu et al, onto ResNetV2, InceptionV3, and MobileNet

### Index

1. [Paper Notes](SENet.md)
2. [Implementation of SE-Inception-V3](SEInceptionV3/SEInceptionV3.py)
3. [Implementation of SE-MobileNet](SEMobileNet/SEMobileNet.py)
4. [Implementation of SE-ResNet](SEResNet/SEResNet.py)


## Usage

### SE-Inception-V3

1. Clone the Repo
2. Run `SEInceptionV3/run.py`

    ```python
    import torch
    from torchinfo import summary
    from SEInceptionV3 import SEInceptionV3

    # init randn tensor

    x = torch.randn( size = (2, 3, 299, 299))

    # init model

    model = SEInceptionV3( reduct_ratio = 16 ) # usign recommended reduction ratio | https://arxiv.org/pdf/1709.01507

    # get model summary and final shape

    summary(model, x.size())
    print(f"\nFinal  Output Size: {model(x).size()}")
    ```

### SE-MobileNEt

1. Clone the Repo
2. Run `SEMobileNet/run.py`
    
    ```python
    import torch
    from torchinfo import summary
    from SEMobileNet import SEMobileNetV1

    # init model -- res mult = .5, depth mult = .75, as example

    model = SEMobileNetV1(rho = .5, alpha = .75)

    # init randn tensor

    x = torch.randn( size = (2, 3, 224, 224))

    # run model, get summary, and final output size

    summary(model, x.size())
    print(f"\nFinal  Output Size: {model(x).size()}")
    ```

### SE-ResNet
1. Clone the Repo
2. Run `SEResNet/run.py`
    
    ```python

    import torch
    from torchinfo import summary
    from SEResNet import SEResNet

    # init randn tensor

    x = torch.randn( size = (2, 3, 224, 224))

    # init model

    model = SEResNet( reduct_ratio = 16) # usign recommended reduction ratio | https://arxiv.org/pdf/1709.01507

    # get model summary and final shape

    summary(model, x.size())
    print(f"\nFinal  Output Size: {model(x).size()}")
    ```

## Citation

```bibtex 
@misc{hu2019squeezeandexcitationnetworks,
      title={Squeeze-and-Excitation Networks}, 
      author={Jie Hu and Li Shen and Samuel Albanie and Gang Sun and Enhua Wu},
      year={2019},
      eprint={1709.01507},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1709.01507}, 
}
```