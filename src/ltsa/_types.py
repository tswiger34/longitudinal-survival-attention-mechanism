"""Module for useful types"""

from typing import TypeAlias, NamedTuple  # noqa
from torch import Tensor


class TranformerLayerOutput(NamedTuple):
    """Output tensors from a single layer in a transformer model

    Attributes:
        feats (Tensor): Output features
        attn_map (Tensor): List of attention mappings from each layer in the transformer
    """

    feats: Tensor
    attn_map: Tensor


class TransformerOutput(NamedTuple):
    """Output tensors from a transformer model

    Attributes:
        feats (Tensor): Output features
        attn_maps (list[Tensor]): List of attention mappings from each layer in the transformer
    """

    feats: Tensor
    attn_maps: list[Tensor]
