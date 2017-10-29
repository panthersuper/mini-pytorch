'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
# Using torchvision, it’s extremely easy to load CIFAR10.

import torchvision
import torchvision.transforms as transforms

import os, datetime
import numpy as np
from DataLoader import *

# Dataset Parameters
batch_size = 120
load_size = 256
fine_size = 32
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.0001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 1
step_display = 50
step_save = 10000
path_save = 'alexnet'
start_from = ''


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 100)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)



net = GoogLeNet()

# send to GPU
net = net.cuda()

print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


# # Initialization
# if len(start_from)>1:
#     saver.restore(sess, start_from)
# else:
#     sess.run(init)

step = 0
running_loss = 0.0

while step < training_iters:
    # Load a batch of training data
    images_batch, labels_batch = loader_train.next_batch(batch_size)

    images_batch = images_batch.reshape(-1, 3,fine_size,fine_size)

    images_batch = torch.from_numpy(images_batch).float()
    labels_batch = torch.from_numpy(labels_batch).type(torch.LongTensor);

    
    # inputs, labels = Variable(images_batch), Variable(labels_batch)

    # send to GPU
    inputs, labels = Variable(images_batch.cuda()), Variable(labels_batch.cuda())

    # zero the parameter gradients
    optimizer.zero_grad()



    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.data[0]
    if step % 100 == 99:    # print every 2000 mini-batches
        print('[%5d] loss: %.3f' %
              (step + 1, loss.data[0]))
        running_loss = 0.0

    step += 1

torch.save(net, "test_checkpoint.pth.tar")

print("Optimization Finished!")
