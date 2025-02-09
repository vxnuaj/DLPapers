import torch
import torch.nn as nn

from blocks import PositionalEncoding, Encoder, Decoder
from Transformers.src.utils import causal_mask, padding_mask

class Transformer(nn.Module):
    
    def __init__(
        self, 
        dropout_p,
        d_model,
        max_len,
        num_heads,
        Y_tokenized_seqs,
        X_tokenized_seqs,
        pad_token_id,
        n,
        device = ('cuda' if torch.cuda.is_available() else 'mps')
        ):
        
        super().__init__()
    
        self.device = device 
        self.Y_tokenized_seqs = Y_tokenized_seqs
        self.X_tokenized_seqs = X_tokenized_seqs
        self.pad_token_id = pad_token_id
        self.n = n # num of total encoder:decoder blocks
        
        self.pe = PositionalEncoding(
            dropout_p = dropout_p,
            d_model = d_model,
            max_len = max_len
        ).to(device)
       
        
        self.encoders = nn.ModuleList(
            [Encoder(
                d_model = d_model,
                num_heads = num_heads
            ) for _ in range(n)]
            ).to(device)
        
        self.decoders = nn.ModuleList(
            [Decoder(
                d_model = d_model,
                num_heads = num_heads
            ) for _ in range(n)]
        ).to(device)
      
       
    def forward(self, x, y):
        
        target_seq_len = y.size(1)
        in_seq_len = target_seq_len # as masked softmax is for self attention, not cross attention.

        csl_msk = causal_mask(
            in_seq_len = in_seq_len, 
            target_seq_len = target_seq_len
        ).to(self.device)
        
        dec_pad_msk = padding_mask(
            tokenized_seqs = self.Y_tokenized_seqs,
            pad_token_id = self.pad_token_id
        ).to(self.device)
       
        enc_pad_msk = padding_mask(
            tokenized_seqs = self.X_tokenized_seqs,
            pad_token_id = self.pad_token_id
        ).to(self.device) 
        
        x_p = x + self.pe(x).to(self.device) # encoder input
        y_p = y + self.pe(y).to(self.device) # decoder input
      
        for n in range(self.n):
            if n == 0:
                enc_out = self.encoders[n](x_p, enc_pad_msk)
                dec_out = self.decoders[n](y_p, enc_out, csl_msk, dec_pad_msk, enc_pad_msk)
            else:
                enc_out = self.encoders[n](enc_out, enc_pad_msk)
                dec_out = self.decoders[n](y_p, enc_out, csl_msk, dec_pad_msk, enc_pad_msk) 
                
        return dec_out
        
        