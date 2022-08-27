# coding: utf-8

import torch
import pickle
from tqdm import tqdm
import os
import random
import json

SOS = 2
EOS = 3


class TopicDataset:
    def __init__(self, data_split, path, shuffle=False, max_len=400, max_target_len=120, use_data_num=-1):
        self.data_split = data_split
        self.data_file = os.path.join(path, 'bart_data_ml%d_mtl%d.pt' % (max_len, max_target_len))
        if not os.path.exists(self.data_file):
            self.data_file = os.path.join(path, 'bart_data_ml%d.pt' % max_len)
            print("Old version, load from: ", self.data_file)

        t = {'train': 0, 'valid': 1, 'test': 2}
        if use_data_num == -1:
            self.data = torch.load(self.data_file)[t[data_split]]
        else:
            self.data = torch.load(self.data_file)[t[data_split]][:use_data_num]
        if shuffle:
            random.shuffle(self.data)
        self.example_num = len(self.data)

    def gen_batch(self):
        for input_ids, attention_mask, labels in self.data:
            yield input_ids, attention_mask, labels

        raise StopIteration


if __name__ == "__main__":
    t = TopicDataset('train',
                     '/data1/tsq/wikicatsum/animal/bart_data')

    for input_ids, attention_mask, labels in t.gen_batch():
        print(input_ids)
        print("mask is ########")
        print(attention_mask)
        print("labels is ########")
        print(labels)
        quit()
