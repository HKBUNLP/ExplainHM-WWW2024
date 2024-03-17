from .base_dataset import JsonDataset
import io
from PIL import Image

class MemeDataset(JsonDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            input_filename = "train.jsonl"
        elif split == "val":
            input_filename = "val.jsonl"
        elif split == "test":
            input_filename = "test.jsonl"

        img_key = "image"
        text_key = "text"
        label_key = "labels"
        rationale_key = "rationale"

        super().__init__(
            *args,
            **kwargs,
            input_filename=input_filename,
            img_key=img_key,
            text_key=text_key,
            label_key=label_key,
            rationale_key=rationale_key,
        )


    def __getitem__(self, index):
        suite = self.get_suite(index)
        return suite
