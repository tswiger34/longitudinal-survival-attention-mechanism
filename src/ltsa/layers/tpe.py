import numpy as np
import torch
from torch import Tensor, nn

### BASED ON https://github.com/bionlplab/longitudinal_transformer_for_survival_analysis/blob/main/src/models.py ###


class TemporalPositionalEncoding(nn.Module):
    """Temporal positional encoding block derived from Holste G et. al.

    If position _i_ is even:
        _TE(v) = sin(v/(10000^(2i/d)))_

    If position _i_ is odd:
        _TE(v) = cos(v/(10000^(2i/d)))_
    """

    def __init__(self, d_model: int, dropout: float = 0.25, max_len: int = 5000):
        """Temporal positional encoding block implementation

        Builds positional indice --> PE denominator term --> applies _sin_ PE to even positions, _cos_ to odd -->
        add PE to model state via a model buffer --> Reorders to batch-first inputs for forward pass

        Args:
            d_model (int): Embedding width
            dropout (float, optional): Dropout probability after adding position info. Defaults to 0.25.
            max_len (int, optional): Maximum sequence length to precompute encodings for. Defaults to 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, ::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.pe = torch.permute(self.pe, (1, 0, 2))

    def forward(self, x: Tensor, rel_times: Tensor) -> Tensor:
        """Forward pass for the TPE block

        Args:
            x (Tensor): Input image tensor
            rel_times (Tensor): Tensor of image acquisition times relative to first image

        Returns:
            Tensor: Tensor output after applying TPE and dropout
        """
        rel_times = rel_times.to(device=x.device, dtype=torch.long)
        x = x + self.pe[0, rel_times, :]
        return self.dropout(x)
