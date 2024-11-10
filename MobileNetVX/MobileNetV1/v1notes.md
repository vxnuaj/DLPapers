# MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

### Abstract & Introduction

- MobileNets use Depthwise Seperable COnvs to build lightweight deep Neural Networks to run on edge devices.
- 2 sets of hyperparamters allow for the model to be constructed to the right size given the computational limits of the given device the model will be operating on.

### Priorwork

- MobileNet focuses on both *size* and *speed*, which proves to be an outlier as other networks purely focused on model *size*.

### MobileNet Architecture

- The model is based on DepthWise Seperable Convolutions
- A Standard Convolution combines the input input channels into as ingle channel, in a single step. The Depthwise convolution splits this into 2 separate layers, via a Depthwise conv an then a pointwise convolution. 
- Depthwise Convs introduce a single Kernel for each channel, such that computational cost reduces by a factor of $M$, where $M$ is the count of channels in a $\mathcal{K}$ for a regular convolution. 
  - But it doesn't combine features across channels, we're only learning representations for individual feature maps -- we need recombination via $1 \times 1$ convolutions

<br/>

- The architecture is built on Depthwise Seperable COnvs besides the first layer which is a full convolution.
  - Each layer goes as : $\text{Conv} \rightarrow \text{BatchNorm} \rightarrow \text{ReLU}$
  - Downsampling is handled by strided convolutions for in the Depthwise Sep. Conv. Layers, $s = 2$     
    - In the first layer as well, handled via the first $3 \times 3$ conv.

<br/> 

- It is'nt enough to purely consider model capacity as a measure for computational cost, ensuring that the model has **fast** inference is also important.
  - Given that MobileNet uses $1 \times 1$ convolutions for most of it's computation, it can be turnt into a $\text{GEMM}$ fairly easily.

- We can introduce a width mulitplier, $\alpha$, which thins the size of MobileNet, such that for any number of input and output channels, $M$ and $N$ respsectively, the input and output channels become $\alpha M$ and $\alpha N$.
  - Done prior to training -- expecting accuracy to decrease for smaller $\alpha$
  - $\alpha \in [0, 1)$
  - This has the effect of automating the proportional scaling down each layer, rather than manually adjusting each layer, in a disproportionate manner
  
- Can also introduce the resolution multiplier, $\rho$, which reduces the spatial dimensions of a given set of input feature maps to the $\ell th$ layer by a factor of $\rho$, where $\rho \in [0, 1)$  
  - only applied to the input layer -- for the original image. 
  - reduces computational cost by $\rho^2$

### Experiments

- Compared to a regular ConvNet, the MobileNet (Depthwise Seperable Convs) architecture only gets $1$% worse compared to the ConvNet, on ImageNet, with a huge amount in Param decrease (only having $11$% of of ops and $14$% of parameters)
