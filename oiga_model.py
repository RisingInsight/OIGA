"""Public interface for the OIGA model.

The full implementation is not part of this review-stage subset. The signatures
below define the public contract used by the training and inference entry points.
"""

import pytorch_lightning as pl

from oiga_constants import CFG_SCALE, NUM_MODALITIES, TEST_SAMPLES_PER_MODALITY


class OIGA(pl.LightningModule):
    """OIGA flow-matching model interface."""

    def __init__(self, model_arch="sit", model_config=None, lr=3e-5,
                 weight_decay=0.01, training_stage=1, cfg_scale=CFG_SCALE, **kwargs):
        super().__init__()
        self.model_arch = model_arch
        self.model_config = model_config or {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_stage = training_stage
        self.cfg_scale = cfg_scale
        self.num_modalities = NUM_MODALITIES
        self.test_samples_per_modality = TEST_SAMPLES_PER_MODALITY

    def forward(self, source, target=None, src_id=None, tgt_id=None,
                text_tokens=None, training=False):
        """Return the predicted target or the training loss dictionary.

        source/target are images; src_id/tgt_id are modality indices; text_tokens encode Y.
        """
        raise NotImplementedError("The full OIGA implementation is not included in this subset.")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("The full OIGA implementation is not included in this subset.")

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError("The full OIGA implementation is not included in this subset.")

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError("The full OIGA implementation is not included in this subset.")

    def configure_optimizers(self):
        raise NotImplementedError("The full OIGA implementation is not included in this subset.")
