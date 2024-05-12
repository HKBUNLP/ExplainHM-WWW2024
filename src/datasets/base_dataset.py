import random
import torch
import io
import pandas as pd
import os
import jsonlines

from PIL import Image
from ..transforms import keys_to_transforms

def jsonl_reader(path):
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            data.append(obj)
    return data

class JsonDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        input_filename,
        transform_keys,
        image_size,
        patch_size,
        img_key,
        text_key,
        label_key,
        tokenizer=None,
        max_text_len=50,
        image_only=False,
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        """
        assert len(transform_keys) >= 1
        super().__init__()
        self.data_dir = data_dir
        self.image_only = image_only
        self.data = jsonl_reader(f"{data_dir}/{input_filename}")
        self.img_key = img_key
        self.text_key = text_key
        self.label_key = label_key
        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.max_text_len = max_text_len
        self.image_size = image_size
        self.patch_size = patch_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data) if self.data else 0
    
    def get_image(self, idx):
        image_features = self.transforms[0](Image.open(f"{self.data_dir}/images/{str(self.data[idx][self.img_key])}")).unsqueeze(0)
        return {
            "image_features": image_features, # [1, 3, H, W]
            "raw_index": idx,
            "img_path": f"{self.data_dir}/images/{str(self.data[idx][self.img_key])}",
            "img_index": self.data[idx]["id"],
        }

    def get_text(self, idx):
        text = str(self.data[idx][self.text_key]).lower()
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "raw_index": idx,
        }
    
    def get_label(self, idx):
        label = self.data[idx][self.label_key][0]
        return {
            "label": label,
            "raw_index": idx,
        }

    def get_suite(self, idx):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(idx))
                if not self.image_only:
                    ret.update(self.get_text(idx))
                    ret.update(self.get_label(idx))
                result = True
            except Exception as e:
                print(f"Error while read file idx {idx} in {self.data_dir}/{str(self.data[idx][self.img_key])} -> {e}")
                idx = random.randint(0, len(self.data) - 1)

        return ret

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        batch_image_features = torch.cat(dict_batch["image_features"], dim=0) # [bs, 3, H, W]
        dict_batch["image_features"] = batch_image_features
        batch_labels = torch.LongTensor(dict_batch["label"])
        dict_batch["labels"] = batch_labels

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]

            # Prepare for text encoder
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_masks"] = attention_mask
        return dict_batch
