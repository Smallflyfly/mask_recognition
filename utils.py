# -*- coding: utf-8 -*-
# @Time    : 2020/5/27 20:11
# @Author  : Fangpf
# @FileName: utils.py

import os
import torch


def save_network(net, epoch):
    filename = 'net_%s.pth' % (epoch+1)
    save_path = os.path.join('./weights', filename)
    torch.save(net.cuda().state_dict(), save_path)
