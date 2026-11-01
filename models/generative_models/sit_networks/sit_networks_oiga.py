"""SiT backbone interface for OIGA."""

import torch.nn as nn

from oiga_constants import (ADAPTER_REDUCTION, CROSS_ATTN_LAYERS, LATENT_CHANNELS,
                         LORA_ALPHA, LORA_RANK, NUM_ADAPTERS, NUM_MODALITIES,
                         PATCH_SIZE, SIT_DEPTH, SIT_HEADS, SIT_WIDTH)


class SiT_OIGA(nn.Module):
    """Scalable Interpolant Transformer with OIGA conditioning."""

    def __init__(self, patch_size=PATCH_SIZE, in_channels=LATENT_CHANNELS,
                 hidden_size=SIT_WIDTH, depth=SIT_DEPTH, num_heads=SIT_HEADS,
                 num_modalities=NUM_MODALITIES, num_adapters=NUM_ADAPTERS,
                 adapter_reduction=ADAPTER_REDUCTION,
                 cross_attn_layers=CROSS_ATTN_LAYERS, lora_rank=LORA_RANK,
                 lora_alpha=LORA_ALPHA, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.num_adapters = num_adapters
        self.adapter_reduction = adapter_reduction
        self.cross_attn_layers = tuple(cross_attn_layers)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

    def forward(self, x, t, src_id, tgt_id, source_latent, text_tokens=None):
        raise NotImplementedError("The full SiT implementation is not included in this subset.")
