<div align = 'center'>
<img src = 'https://www.researchgate.net/publication/368539704/figure/fig3/AS:11431281143702109@1681264260877/Block-diagram-of-the-MobileNetV2-architecture.png'/>
</div>

# MobileNetVX

Notes and PyTorch Implementations of MobileNetV1 & MobileNetV2.

## Index

1. [MobileNetV1](MobileNetV1)
   1. [Implementation](MobileNetV1/mobilenetv1.py)
   2. [Notes](MobileNetV1/v1notes.md)
2. [MobileNetV2](MobileNetV2)
   1. [Implementation]
   2. [Notes]

## MobileNetV1

Implementation of MobileNetV1, proposed on *"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"*, by Howard et al.

### Usage

1. Clone the Repo
2. Run `runv1.py`

   ```python
   import torch
   from torchinfo import summary
   from mobilenetv1 import MobileNetV1

   # init model -- res mult = .5, depth mult = .75, as example

   model = MobileNetV1(rho = .5, alpha = .75)

   # init randn tensor

   x = torch.randn( size = (2, 3, 224, 224))

   # run model, get summary, and final output size

   summary(model, x.size())
   print(f"\nFinal  Output Size: {model(x).size()}")
   ```

## MobileNetV2 (TODO, INCOMPLETE)

Implementation of MobileNetV2, proposed on *"MobileNetV2: Inverted Residuals and Linear Bottlenecks"*, by Sandler et al.

### Usage

1. Clone the Repo
2. Run `runv2.py`

   ```python
   import torch
   from torchinfo import summary
   from mobilenetv2 import MobileNetV2

   # init model

   model = MobileNetV2()

   # init randn tensor

   x = torch.randn( size = (2, 3, 224, 224))

   # run model, get summary, and final output size

   summary(model, x.size())
   print(f"\nFinal  Output Size: {model(x).size()}")
   ```


## Citations

```bibtex
@misc{howard2017mobilenetsefficientconvolutionalneural,
      title={MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications}, 
      author={Andrew G. Howard and Menglong Zhu and Bo Chen and Dmitry Kalenichenko and Weijun Wang and Tobias Weyand and Marco Andreetto and Hartwig Adam},
      year={2017},
      eprint={1704.04861},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1704.04861}, 
}

@misc{sandler2019mobilenetv2invertedresidualslinear,
      title={MobileNetV2: Inverted Residuals and Linear Bottlenecks}, 
      author={Mark Sandler and Andrew Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen},
      year={2019},
      eprint={1801.04381},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1801.04381}, 
}
```
