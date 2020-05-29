# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 18:40
# @Author  : Fangpf
# @FileName: make_data.py

import os

ROOT_PATH = './data'


def make_file(path):
    folders = os.listdir(path)
    with open(os.path.join(path, 'train.txt'), 'wt') as f:
        for folder in folders:
            images = os.listdir(os.path.join(path, folder))
            if folder == 'unmasked_images':
                masked = 0
            else:
                masked = 1
            for image in images:
                f.write(os.path.join(path, folder, image) + ' ' + str(masked) + '\n')
    print('done!')


if __name__ == '__main__':
    make_file(ROOT_PATH)
