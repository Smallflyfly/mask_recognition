# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 15:39
# @Author  : Fangpf
# @FileName: demo.py

import os
from PIL import Image
from myresnet import resnet50
from torchvision import transforms as T
import torch
import cv2
import torch.nn as nn


def demo(image, weight):
    model = resnet50(num_classes=2)
    model.eval()
    transform = T.Compose([
        T.Resize(size=(256, 256)),
        T.ToTensor(),
        T.Normalize([0.56687369, 0.44000871, 0.39886727], [0.2415682, 0.2131414, 0.19494878])
    ])
    soft_max = nn.Softmax()
    # im = Image.open(image)
    im = cv2.imread(image)
    # opencv BGR --> RGB
    im = im[:, :, ::-1]
    im = transform(im)
    im = im.reshape(1, im.shape[0], im.shape[1], im.shape[2])
    if torch.cuda.is_available():
        model.cuda()
        im.cuda()
    out = model(im)
    out = soft_max(out)
    prob = out.max(1)
    print(out)
    print(prob)


if __name__ == '__main__':
    path = './demo'
    images = os.listdir(path)
    weight = './weights/*.pth'
    for image in images:
        demo(os.path.join(path, image), weight)