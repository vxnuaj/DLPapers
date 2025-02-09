# Attention is All You Need.

Notes and PyTorch implementation of the Transformer Architecture, proposed by Vaswani et al in *"[Attention is all you need](https://arxiv.org/pdf/1706.03762)".*

### Index:

1. [Paper Notes](notes.md)
2. [Implementation](src/)
   
## Usage

1. Clone the Repo
2. Run [`main.py`](src/main.py)

```python
import torch
from model import Transformer

device = ('cuda' if torch.cuda.is_available() else 'mps')

X = torch.randn(size = (5, 100, 512)).to(device)
Y = torch.randn(size = (5, 90, 512)).to(device)
Y_tokenized_seqs = torch.randint(low=1, high=13000, size=(Y.size(0), Y.size(1))).to(device)
X_tokenized_seqs = torch.randint(low = 1, high = 13000, size = (X.size(0), X.size(1))).to(device)

padding_token = 0
num_padding_tokens = 5
Y_seq_len = Y_tokenized_seqs.size(1)
X_seq_len = X_tokenized_seqs.size(1)

Y_random_positions = torch.randint(0, Y_seq_len, size=(Y_tokenized_seqs.size(0), num_padding_tokens)).to(device)
Y_tokenized_seqs[torch.arange(Y_tokenized_seqs.size(0)).unsqueeze(1), Y_random_positions] = padding_token

X_random_positions = torch.randint(0, X_seq_len, size=(Y_tokenized_seqs.size(0), num_padding_tokens)).to(device)
X_tokenized_seqs[torch.arange(X_tokenized_seqs.size(0)).unsqueeze(1), X_random_positions] = padding_token

d_model = 512
dropout_p = .1
max_len = 100
num_heads = 2
in_seq_len = 100
target_seq_len = 100
n = 5

model = Transformer(
    dropout_p = dropout_p,
    d_model = d_model,
    max_len = max_len,
    num_heads = num_heads,
    X_tokenized_seqs = X_tokenized_seqs,
    Y_tokenized_seqs = Y_tokenized_seqs,
    pad_token_id = padding_token,
    n = n
).to(device)

x = model(X, Y)

print(x.size()) # batch_size x target_seq_len x d_model
```