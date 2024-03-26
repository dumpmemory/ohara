import torch
def build_mask(seq_len, sliding_window_attention=False, window_size=1):
    mask = torch.full((seq_len, seq_len), float("-inf"))
    
    assert window_size != 0 , "window_size cannot be 0"
    if not sliding_window_attention:
        window_size = seq_len

    row_indices = torch.arange(seq_len).unsqueeze(-1)
    col_indices = torch.arange(seq_len)
    distance = row_indices - col_indices

    mask[(distance >= 0) & (distance <= (window_size-1))] = 0

    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

if __name__ == "__main__":
    seq_len = 7  
    mask = build_mask(seq_len, sliding_window_attention=True, window_size=1)
    print(mask)