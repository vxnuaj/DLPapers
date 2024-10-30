# Going Deeper with Convolutions

**Initial Thoughts**: 

It's interesting to see how Inception uses multi-sized $\mathcal{K}$. For intuition, I think it'll help a given layer learn representations of different sized receptive fields, such that subsequent layers have "sight" into different ways of how the spatial structure is defined for a given input feature.

You combine this with more inception blocks, and hierarchically, each layer gets a more varying / descriptive insight into how an input feature map is constructed, spatially.

You're able to capture **both** fine-grained details and broader higher-level larger structures of a feature map with the inception module, building multi-scale features.

At a given $l$ layer, it has multi-sized receptive fields, depending on the branch you consider as each branch has different sized $\mathcal{K}$

### Definitions

**NiN:** *Network in Network*, by Lin et al

**Polyak Averaging**: Track an exponentially weighted average of the weights and use those weights during inference / testing.

```math

\bar{\Theta}_t = \alpha\bar{\Theta}_{t - 1} - (1 - \alpha)\Theta_{t}

```

where $\Theta_t$ is the parameter in question, $\bar{\Theta}_t$ is the averaged parameter, and $t$ is the current iteration.

The averaged weights are used define the model during inference. If the model is oscillating between the parallel points, surrounding the optima, applying a weighted average can help move the parameters to the optimal set of weights for the model.


### Introduction

- Progress in Computer Vision (2012 - 2014) hasn't been purely bigger data and GPUs but also more efficient yet accurate algorithms
- With on-device inference, on phones and other embedded computing, power and memory efficiency as extremely important
- Inception was built with a computational budget of 1.5B mulitply-adds during inference.
- Outperforms SOTA (as of 2014) in ILSVRC.

### Related Work

- Inception is a 22-layer model
- Inception makes use of $1 \times 1$ convolutions, to perform channel-wise parametric pooling, such that for the next layer, computational complexity is reduced while still retaining valuable information regarding relationships of a given feature at the $i, j$ position of the feature maps, across channels.
  - Allows for increasing depth and larger receptive field of the model.
- For Inception's object detection sumission to ILSVRC, they used Selective Search, just like R-CNN to generate bounding box region proposals

### Motivation & Considerations

- The best means to improve model performance is to increase their size, both width and depth, primarily depth.
  - Unfortunately this increases computational complexity.
  - This also increases the need for a larger dataset to prevent overfitting, which can be expensive to gather.
- A means to reduce network size is to introduce sparsity, by conjecturing that a generalized accurate hypothesis of the spatial structure can be build with smaller sets of parameters in $\mathcal{K}$. (This is why Inception uses multiple branches of different sizes, *why have a single branch of $5 \times 5$ convolutions when you can use two branches of $1 \times 1$ and $5 \times 5$, with the same amount of output channels as the former after concatenation?* **You're still able to learn important features despite having less parameters (given a smaller $1 \times 1$ conv).**
- To construct the optimal construction (without redundant weight sizes), you can analyze the correlation statistics of different regions in a given input, $X$, and construct kernels of sizes related to the dimensions of highly correlated regions.
- For earlier layers, a given $\mathcal{K}$ is trained to detect low-level features, such as edges, corners, etc. These low level features aren't abstract and are more locally defined, hence a given activation from $\mathcal{K}$, will likely be similar or correlated to other activations accross channels of the given activation of $\mathcal{K}$. 
  - Such then is the case that you can use as $1 \times 1$ convolution to sumamrize the highly correlated activations, in a parameterized manner, with a non-linearity to retain the valuable information of the feature map at the specific $i, j$ position.
  - As you go deeper into the convnet, the extracted edges and corners will begin to represent more spatially spread out clusters of important features (given by correlation statistics) such that you'll be able to learn them through larger sizes of $\mathcal{K}$, such as $5 \times 5$ convolutions. 
  - Thereby, as you get deeper into the model, you want to be able to increase the ratio of larger sized $\mathcal{K}$ ($3 \times 3$ & $5 \times 5$) relative to the $1 \times 1$ $\mathcal{K}$.

- The issue with increasing the ratio of $3 \times 3$ and $5 \times 5$ convolutions as we get deeper is that the number of output channels can continue to grow large. So drawing inpsiration from NiN, they use $1 \times 1$ convolutions to perform depthwise parametric pooling, such that they compress the amount of channels but aim to continue retaining information (via parameters and a non-linearity after each $1 \times 1$ conv.).

### GoogLeNet

<div align = 'center'>
<img src = 'https://media.geeksforgeeks.org/wp-content/uploads/20200429201421/Inception-layer-by-layer.PNG' width = 500>
<img src = 'https://media.geeksforgeeks.org/wp-content/uploads/20200429201549/Inceptionv1_architecture.png'>
</div>
<br>


> #3x3 reduce and #5x5 reduce columns indicate the amount of 1x1 conv filters used prior to a given convolution (either 3x3 of 5x5, respective to the column label.)

- All convolutions use $\text{ReLU}$ including those within the inception modules
- The network takes in $224 \times 224$ images.
- They use linear layer after global average pooling isntead of $1 \times 1$ convolutions, simply for purposes of allowing fine tuning, but still needed the use of dropout to avoid overfitting.
- The large depth of the network introduced the risk of vanishing gradients, therefore they used auxiliary classifiers after certain stages of the architecutre.
  - The auxiliary classifiers were used after the $3$rd and $6$th inception module.
  - They were comprised of:
    1. $5 \times 5$ average pooling layers with stride $3$
    2. A $1 \times 1$ convolution with $128$ filters for channel reduction
    3. $\text{ReLU}$ after each conv and fc layer besides output
    4. an FC layer with $1024$ neurons and Dropout with $p = .70$.
    5. an FC layer with $1000$ neurons and Softmax Activation
- Momentum with $\beta = .9$.
- Decreased learning rate by 4% every 8 epochs.
- Polyak averaging was used for the final model used at inference.
  
### Key Insights / Closing Thoughts

Earlier layers should be optimzed to learn local features, wuch as edge and corners, and slowly increase the receptive field over time.

- The deeper your model is, the more parameter efficient and accurate it becomes (hierarchical non-linear feature extraction)
- The correlation statistics of a given input image denote that low-level features tend to be concentrated in local regions (edges and corners are locally correlated and localized phenomena $\rightarrow$, they can be learnt via $1 \times 1$ convs)
  - Your receptive field remains small when you initially apply the $1 \times 1$ convs.
- As your model learns local features, it will construct feature maps where the **important** edges and features are more well defined such that they begin to define larger correlated regiosn of features, that can be learnt via large convolutions, such as $3 \times 3$ or $5 \times 5$ convolutions.
  - The receptive field begins to slowly grow as we continue to introduce the larger convs.