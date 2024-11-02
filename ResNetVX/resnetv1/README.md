## Deep Residual Learning for Image Recognition

### Abstract

- Deep Neural Networks are hard to train

    > Implying Vanishing Gradients

- In ResNet, layers are reformulated as residual functions with reference to the input layers.

    > Residual Functions refer to a functoin that represents the difference between the observation and the prediction.

### Introduction

- As models get deeper, $∂$'s $\rightarrow 0$, such that the deeper the model is, the smaller a given $∂$ will be at earlier layers, due to the nature of the chain rule. 
  - Thereby, given the update rule $\Theta = \Theta - \alpha * ∂$, for a deeper network it is possible that we end up with larger training error due to the inability to converge onto an optima as the $∂$'s are too small for meaningful updates.

<div align = 'center'>  
<img width = 500 src = err.png>
</div>

- Residual Connections **beat** deeper models, deeper models didn't have comparable training error to models with residual connections.
- Given the target, $\mathcal{H}(x)$, the model learns the residual, $\mathcal{F}(x):= \mathcal{H}(x) - x$ and then constructs an output as $\hat{\mathcal{H}}(x) = \mathcal{F}(x) + x$.

### Deep Residual Learning

- Residual Connections are simply $y = f(x) + x$, where $f(x)$ represents a multi-layer neural network (feed forward or convolutional).
- The vanilla Residual Connection works when the input is same size as the output
  - If the connection connects an $x$ that is differemt dimension to to $f(x)$, you can pad $f(x)$.
  - Or you can create a weighted matrix, $W_s$ such that $y = f(x) + W_sx$ such that $W_s$ performs the linear transformation of $x$ into $\mathbb{R}^n$, where $n$ is the dimensionality of $f(x)$.
    - Expressed via convolutions, this can be equivalent to $1 \times 1$ convolutions with $\mathcal{K}$, where we decrease the channel size to the desired channel count.
    - Then $W_s$ or $\mathcal{K}$ can be learnt parameters.
    - In the ResNet shown in the paper, their $1 \times 1$ convolutions were applied with a stride of $2$ such that they effectively reduce the dimensions of the model to properly fit the desired size.
    - They use $\text{PCA}$ pixel based RGB augmentation (see [here](https://www.vxnuaj.com/blog/PCAaug))
    - They use Batchnorm after each convolution prior to each activation and after eacih convolution.
    - They use Xavier Initialization.