"""CLIP text encoder with LoRA interface."""

import torch.nn as nn


class CLIPTextEncoderLoRA(nn.Module):
    """Frozen CLIP text encoder with low-rank adaptation."""

    def __init__(self, model_name="ViT-L/14", rank=16, alpha=16):
        super().__init__()
        self.model_name = model_name
        self.rank = rank
        self.alpha = alpha

    def forward(self, text_tokens):
        raise NotImplementedError("The full CLIP-LoRA implementation is not included in this subset.")
