import torch

def causal_mask(
    in_seq_len, 
    target_seq_len,
    ):
 
    '''
    
    returns size (target_seq_len, in_seq_len) 
    
    '''
  
    x = torch.ones(
        size = (
            target_seq_len,
            in_seq_len
        )
    )
    
    mask = torch.tril(
        input = x,
        diagonal = 0
    ) 
    
    return mask.bool()

def padding_mask(
    tokenized_seqs,
    pad_token_id 
    ):
 
    '''
    returns size (batch_size, seq_len) 
    '''
  
    mask = (tokenized_seqs != pad_token_id).long()
    
    return mask.bool()