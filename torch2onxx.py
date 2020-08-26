# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 10:39
# @Author  : Fangpf
# @FileName: torch2onxx.py.py
import torch

from myresnet import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './weights/net_15.pth'


def torch2onxx():
    onxx_model = "mask.onxx"
    dummy_input = torch.randn(1, 3, 256, 256, requires_grad=True)
    model = resnet50(num_classes=2)
    model = model.load_state_dict(torch.load(model_path))
    torch.onnx.export(model,
                      dummy_input,
                      onxx_model, verbose=False,
                      training=False, do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output']
                      )