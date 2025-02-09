import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import MultiheadAttention

import math

class PositionalEncoding(nn.Module):
    
    def __init__(self, dropout_p: float, d_model: int, max_len: int):
        super().__init__()
       
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(size=(max_len, d_model))  
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 

        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:, :x.size(1)] 
        return pe

class QKV(nn.Module):

    '''
    in_features: the embedding size
    
    out_features: the output size of the projection into the Query Space, Value Space, and Key Space. equivalent for all. 
    '''

    def __init__(self, d_model, in_:bool = True):
        
        super().__init__()
      
        assert isinstance(in_, bool) 
      
        self.in_ = in_ 
       
        if self.in_:

            self.linearQ = nn.Linear(
                in_features = d_model,
                out_features = d_model
            )
            
            self.linearK = nn.Linear(
                in_features = d_model,
                out_features = d_model
            )
            
            self.linearV = nn.Linear(
                in_features = d_model,
                out_features = d_model
            )
        
        self.linearQA = nn.Linear(
            in_features = d_model,
            out_features = d_model
        )
        
        self.linearKA = nn.Linear(
            in_features = d_model,
            out_features = d_model,
        )
        
        self.linearVA = nn.Linear(
            in_features = d_model,
            out_features = d_model
        )
        
    def forward(self, x):
       
        '''
        Input dims: (BatchSize, SequenceLen, d_model)
        Output dims: (BatchSize, SequenceLen, AttentionSpace)
        '''
       
        if self.in_: 
        
            Q = self.linearQ(x)
            K = self.linearK(x)
            V = self.linearV(x)
       
            QA = self.linearQA(Q) 
            KA = self.linearKA(K)
            VA = self.linearVA(V)
       
        else:
           
            assert isinstance(x, tuple), ValueError('if self.in_ == False, then x must be a tuple -- denotes extraction of QKV for cross attention')
     
            Q, K, V = x 
            
            QA = self.linearQA(Q) 
            KA = self.linearKA(K)
            VA = self.linearVA(V)
        
        return QA, KA, VA

class PositionWiseNN(nn.Module):
    
    def __init__(
        self,
        d_model,
        ):
        
        super().__init__()
       
        n_hidden = 4 * d_model 
        
        self.linear_in = nn.Linear(
            in_features = d_model,
            out_features  = n_hidden
        )
        
        self.linear_out = nn.Linear(
            in_features = n_hidden,
            out_features = d_model
            
        )
        
    def forward(self, x):
        
        x = F.relu(self.linear_in(x))
        x = self.linear_out(x)
        
        return x

class Encoder(nn.Module):
    
    def __init__(
        self, 
        d_model,
        num_heads
        ):
        
        super().__init__()
      
        self.qkv = QKV(
            d_model = d_model, 
            )
       
        self.multihead_attention = MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            batch_first = True
        )
        
        self.layer_norm1 = nn.LayerNorm(
            normalized_shape = d_model
        )
        
        self.positionNN = PositionWiseNN(
            d_model = d_model
        )
       
        self.layer_norm2 = nn.LayerNorm(
            normalized_shape = d_model
        ) 
        
    def forward(self, x, pad_msk):
       
        '''
        Input dims: (BatchSize, SequenceLen, d_model)\n
        Output dims: (BatchSize, SequenceLen, d_model)
        '''
        
        qa, ka, va = self.qkv(x)
        h_self_attn, _ = self.multihead_attention(qa, ka, va, key_padding_mask = pad_msk)
        h_res = self.layer_norm1(h_self_attn + x)
        h_pos = self.positionNN(h_res)
        enc_out = self.layer_norm2(h_pos + h_res)
        
        return enc_out
       
        
class Decoder(nn.Module):
    
    def __init__(
        self,
        d_model,
        num_heads,
        ):
       
        # NOTE \n
        # csl_msk_val = -1e30 assumes FP64, common on high end nvidia chips. \n
        # change to lower val (eg 1e-9) if on FP < FP64 \n
        
        super().__init__()
     
        self.qkv1 = QKV(
            d_model = d_model
        ) 
        
        self.masked_multihead_attention = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            batch_first = True
        )
        
        self.layernorm1 = nn.LayerNorm(
            normalized_shape = d_model
        )
       
        
        self.qkv2 = QKV(
            d_model = d_model,
            in_ = False
        )
       
        self.cross_multihead_attention = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            batch_first = True
        )
        
        self.layernorm2 = nn.LayerNorm(
            normalized_shape = d_model
        )
        
        self.positionNN = PositionWiseNN(
            d_model = d_model
            )
        
        self.layernorm3 = nn.LayerNorm(
            normalized_shape = d_model
        )
        
    def forward(self, y, enc_out, csl_msk, pad_msk, enc_pad_msk):
       
        qa, ka, va = self.qkv1(y)
       
        h_masked_attn, _ = self.masked_multihead_attention(
            qa, 
            ka, 
            va, 
            attn_mask = csl_msk,
            key_padding_mask = pad_msk
            )

        h_res = self.layernorm1(h_masked_attn + y)
        
        qa, ka, va = self.qkv2(
            x = (
                h_res, 
                enc_out,
                enc_out
                )
            )

        h_cross_attn, _ = self.cross_multihead_attention(qa, ka, va, key_padding_mask = enc_pad_msk)
        h_res2 = self.layernorm2(h_cross_attn + h_res)
        h_pos = self.positionNN(h_res2)
        dec_out  = self.layernorm3(h_pos + h_res2)
      
        return dec_out