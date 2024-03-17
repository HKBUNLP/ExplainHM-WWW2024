from ..datasets import MemeDataset
from .datamodule_base import BaseDataModule


class MemeDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MemeDataset

    @property
    def dataset_cls_no_false(self):
        return MemeDataset

    @property
    def dataset_name(self):
        return "meme"
