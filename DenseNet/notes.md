# Densely Connected Convolutional Networks

### Introduction

- We propose a *densely connected network*, of which for the $L$th layer, every single preceding $L-n$ layer is connected to it.
- DenseNet requires less parameters than ResNets.
  - ResNets were shown to have redundant layers, as in Stochastic Depth Networks, you could easily drop layers of ResNet randomly and still achieve good results.
  - Thereby, a DenseNet can have less parameters, and given that it concatenates information from a previous layer to the next, within each DenseBlock, we don't need as many parameters to learn meaningful features.

### DenseNets

- ResNets are typically formulated as $x_{\ell} = H(x_{\ell}) + x_{\ell}$, with the advantage of having $x_{\ell}$ as an identity transformation which allows for more effective gradient flow to earlier layers (see [densegrad.md](densegrad.md))
- But the $I$ transformation for the given residual connection may impede information flow within the network, as it's a simple summation to $H$, not allowing feature re-use but rather purely learning the residuals.
- BottleNeck layers -- given that for a DenseBlock, there may be a significant amount of input features, at each layer in the DenseBlock, we can reduce to $4k$ feature maps via $1 \times 1$ convolutions and then to $k$ with $3 \times 3$ convolutions.
- Transition Layers -- given that at each DenseBlock, we want to reduce the spatial dimensions but increase the count of output channels, to allow for efficient concatenation at each layer, the input to the denseblock must match dimensionality in the channel-dimensions and $H \times W$ dimensions.
  - So we introduce a $1 \times 1$ convolution and then $2 \times 2$ AvgPooling with $s = 2$ to reduce the spatial and channel dimensions to match the dimensionality of the outputs in the denseblock.
  - This is implemented with BN-ReLU-Conv and is referred to as DenseNet-B
- To compress the model, they reduce the feature-maps at transition layers.
  - If a DenseBlock has $m$ feature maps as their output, the following transition layer generates $\theta m$ feature maps, where $\theta$ is the compression factor, $\in 0 < \theta ≤ 1$
  - This is referred as DenseNet-C
  - The model that puts compression at bottleneck and transition layers is DenseNet-BC.
  
### Discussion

- DenseNet has improved model compactness alongside improved gradient flow (via deep supervision), that allows for better learning while still having a compact model size.