<div align = 'center'>
<img src = 'https://miro.medium.com/v2/resize:fit:2000/1*UItPkoIvPZR5iXgzVgap6g.png'>
</div>

# ResNeXt

Implementation of ResNeXt, proposed on ["Aggregated Residual Transformations for Deep Neural Networks"](https://arxiv.org/abs/1611.05431) by Xie et al.

**Index**

- [Implementation](resnext.py)
- [Notes](notes.md)

### Usage

1. Clone the Repo
2. Run `run.py`

    ```python
    import torch
    from torchinfo import summary
    from resnext import ResNeXt50

    # init randn tensor

    x = torch.randn(size = (2, 3, 224, 224))

    # init model & get summary

    model = ResNeXt50()

    summary(model, input_size = x.size())
    ```

### Citations

```bibtex

@misc{xie2017aggregatedresidualtransformationsdeep,
      title={Aggregated Residual Transformations for Deep Neural Networks}, 
      author={Saining Xie and Ross Girshick and Piotr Dollár and Zhuowen Tu and Kaiming He},
      year={2017},
      eprint={1611.05431},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1611.05431}, 
}


```
