# -*- coding: utf-8 -*-
# @Time    : 2020/5/27 19:17
# @Author  : Fangpf
# @FileName: train.py.py

from myresnet import resnet50
from dataset import MyDataset
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import tensorboardX as tb
import logging as log
from utils import save_network


def train(net, data_file, label_file, epochs, lr):
    transforms = T.Compose([
        T.Resize(size=(256, 256)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize([0, 0, 0], [0, 0, 0])
    ])
    dataset = MyDataset(data_file, label_file, transforms)
    model = net
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_func = nn.CrossEntropyLoss()
    model.train(True)
    num_epochs = 0
    writer = tb.SummaryWriter()
    for epoch in range(epochs):
        for index, data in enumerate(data_loader):
            im, label = data
            label = label.long()
            if torch.cuda.is_available():
                im = im.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            out = model(im)
            loss = loss_func(out)
            loss.backward()
            optimizer.step()
            num_epochs += 1
            writer.add_scalar('loss', loss, num_epochs)
            if index % 10 == 0 or index == len(data_loader)-1:
                log.info('{} / {}: {} / {} -----------> loss: {}'.format(epoch+1, epochs, index+1, len(data_loader), loss))
        if epoch+1 % 5 == 0:
            save_network(net, epoch+1)

    writer.close()


if __name__ == '__main__':
    net = resnet50(num_classes=2)
    epoch = 50
    lr = 0.005
    data_file = './data/train.txt'
    label_file = './data/label.txt'
    train(net, data_file, label_file, epoch, lr)