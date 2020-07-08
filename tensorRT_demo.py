# -*- coding: utf-8 -*-
# @Time    : 20-7-8 下午7:32
# @Author  : smallflyfly
# @FileName: tensorRT_demo.py
import time

import cv2

import torch
from PIL import Image
from torch import nn
from torch.backends import cudnn
from torch2trt import torch2trt
from torchvision.transforms import transforms as T

from myresnet import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)
cudnn.benchmark = True
transform = T.Compose([
        T.Resize(size=(256, 256)),
        T.ToTensor(),
        T.Normalize([0.56687369, 0.44000871, 0.39886727], [0.2415682, 0.2131414, 0.19494878])
    ])
soft_max = nn.Softmax()
masked_dic = {0:"unmasked", 1:"masked"}

def main(image, weight):
    model = resnet50(num_classes=2)
    model = model.load_state_dict(torch.load(weight))
    model.eval()
    model = model.to(device)
    im = cv2.imread(image)
    im = im[:,:,::-1]
    im = Image.fromarray(cv2.cvrt(im, cv2.COLOR_BGR2RGB))
    im = transform(im)
    im = torch.from_numpy(im).unsqueeze(0)
    im = im.to(device)
    tic = time.time()
    model_trt = torch2trt(model, [im], fp16_mode=True, max_workspace_size=1000)
    print('net forward time: {:.4f}'.format(time.time() - tic))
    out = model_trt(im)
    out = soft_max(out)
    y = torch.argmax(out)
    print(masked_dic[y])


if __name__ == '__main__':
    img = './test.jpg'
    weight = './weight/'
    main(img, weight)