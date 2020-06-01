# -*- coding: utf-8 -*-
# @Time    : 2020/5/27 18:57
# @Author  : Fangpf
# @FileName: dataset.py

from torch.utils import data
from PIL import Image
import PIL
import numpy as np


class MyDataset(data.Dataset):

    def __init__(self, data_file, transforms=None, mode='train'):
        super(MyDataset, self).__init__()
        self._data_file = data_file
        self._transforms = transforms
        self._mode = mode
        self._data = []
        self._label = []

        with open(self._data_file) as df:
            lines = df.readlines()
            for line in lines:
                self._data.append(line.strip().split()[0])
                self._label.append(line.strip().split()[1])

    def __getitem__(self, index):
        im = Image.open(self._data[index])
        if self._transforms:
            im = self._transforms(im)
        label = self._label[index]
        return im, label

    def __len__(self):
        return len(self._data)

