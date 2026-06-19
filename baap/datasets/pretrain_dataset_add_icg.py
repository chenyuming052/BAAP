import os
import pickle
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
# from nltk.tokenize import RegexpTokenizers
from tqdm import tqdm
from transformers import BertTokenizer
from random import shuffle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class ICGPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=112, dataset_path=None):
        super().__init__()
        dataset_path = dataset_path or os.environ.get(
            "BAAP_ICG_CXR_CSV",
            "data/icg-cxr-full/chexpertplus_train.csv",
        )

        if not os.path.exists(dataset_path):
            raise RuntimeError(f"{dataset_path} does not exist!")

        # self.df = pd.read_csv(MIMIC_CXR_IT_CSV).astype(str)

        self.transform = transform
        self.imsize = imsize

        self.tokenizer = BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words
        self.df=pd.read_csv(dataset_path).astype(str)

        if split=="train":
            temporal_json = os.environ.get(
                "BAAP_ICG_CXR_JSON",
                "data/icg-cxr-full/chexpertplus_train.json",
            )
            with open(temporal_json, 'r') as json_file:
                self.temporal = json.loads(json.load(json_file))


if __name__ == "__main__":
    from baap.datasets.transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = ICGPretrainingDataset(split="valid", transform=transform)

    # data = dataset[0]
    # print(dataset[0])
    # print(dataset[1])
    # print(dataset[2])
    # print(dataset[3])
    # print(dataset[4])
    # print(len(dataset))  # 96011

    # pos_query = [
    #     'Findings consistent with pneumonia',
    #     'Findings suggesting pneumonia',
    # ]
    # print(dataset.get_caption(pos_query))
