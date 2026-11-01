"""FLAIR-HUB data module interface for OIGA."""

import pytorch_lightning as pl

from oiga_constants import IMAGE_SIZE, MODALITIES, TEST_SAMPLES_PER_MODALITY


class FLAIRDataModule(pl.LightningDataModule):
    """Data module returning source, target, source ID, target ID, and text tokens."""

    def __init__(self, datasets_folder, train_batch_size=8, test_batch_size=4,
                 train_image_size=(IMAGE_SIZE, IMAGE_SIZE), num_workers=4,
                 num_test_samples=TEST_SAMPLES_PER_MODALITY, **kwargs):
        super().__init__()
        self.datasets_folder = datasets_folder
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_image_size = train_image_size
        self.num_workers = num_workers
        self.num_test_samples = num_test_samples
        self.modalities = MODALITIES

    def setup(self, stage=None):
        raise NotImplementedError("The full data pipeline is not included in this subset.")

    def train_dataloader(self):
        raise NotImplementedError("The full data pipeline is not included in this subset.")

    def val_dataloader(self):
        raise NotImplementedError("The full data pipeline is not included in this subset.")

    def test_dataloader(self):
        raise NotImplementedError("The full data pipeline is not included in this subset.")
