# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 18:59
# @Author  : Fangpf
# @FileName: image_normalization.py

import os
import numpy as np
import cv2

ROOT_PATH = './data'


def cal_mean_std(path):
    folders = ['masked_images', 'unmasked_images']
    mean, std = None, None
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        for image in images:
            im = cv2.imread(os.path.join(path, folder, image))
            im = im[:, :, ::-1]
            im = im / 255.
            # im = im.reshape(1, im.shape[0], im.shape[1], im.shape[2])
            if mean is None and std is None:
                mean, std = cv2.meanStdDev(im)
            else:
                mean_, std_ = cv2.meanStdDev(im)
                mean_stack = np.stack((mean, mean_), axis=0)
                std_stack = np.stack((std, std_), axis=0)
                mean = np.mean(mean_stack, axis=0)
                std = np.mean(std_stack, axis=0)
    return mean.reshape((1, 3))[0], std.reshape((1, 3))[0]


if __name__ == '__main__':
    mean, std = cal_mean_std(ROOT_PATH)
    print(mean, std)

    # im = cv2.imread('./naza.jpg')
    # im = im[:, :, ::-1] / [255., 255., 255.]
    # mean, std = cv2.meanStdDev(im)
