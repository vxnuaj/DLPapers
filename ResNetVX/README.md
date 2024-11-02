<div align = 'center'>
<img src = 'https://media.geeksforgeeks.org/wp-content/uploads/20200424011138/ResNet.PNG'>
</div>

# ResNet VX

Notes and Implementation of ResNetV1 and ResNetV2.

**Index**

1. [ResNetV1](ResNetv1)
   1. [Notes](resnetv1/readme.md)
   2. [Implementation](resnetv1/resnetv1.py)
2. [ResNetV2](resnetv2)
   1. [Notes](resnetv2/README.md)
   2. [Implementation](resnetv2/resnetv2.py) 

## ResNet V1

### Usage

1. Clone the Repo
2. Install pytorch and torchinfo
   ```zsh
    pip install torch torchvision
    pip install torchinfo
   ```
3. Run `run.py` variants

    `run.py`

    ```python
    import torch
    from torchinfo import summary
    from resnetv1 import ResNetV1

    # init random tensor

    x = torch.randn(2, 3, 224, 224)

    # init model & get summary

    model = ResNetV1()

    summary(model, input_size = x.size())
    ```

## ResNet V2

### Usage

1. Clone the Repo
2. Install pytorch and torchinfo
   ```zsh
    pip install torch torchvision
    pip install torchinfo
   ```
3. Run `run.py` variants

    `run.py`

    ```python
    import torch
    from torchinfo import summary
    from resnetv2 import ResNetV2

    # init random tensor

    x = torch.randn(2, 3, 224, 224)

    # init model & get summary

    model = ResNetV2()

    summary(model, input_size = x.size())
    ```


### Citations

```bibtex

@misc{he2015deepresiduallearningimage,
      title={Deep Residual Learning for Image Recognition}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1512.03385}, 
}

@misc{he2016identitymappingsdeepresidual,
      title={Identity Mappings in Deep Residual Networks}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2016},
      eprint={1603.05027},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1603.05027}, 
}

```