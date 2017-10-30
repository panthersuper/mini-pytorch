from TestPrint import TestPrint
import torch
import os
import sys
sys.path.insert(0, '../')
import wideresnet
import numpy as np

load_size = 128
fine_size = 112
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

model  = wideresnet.resnet18(num_classes=100)
model = torch.nn.DataParallel(model).cuda()

# print(model.state_dict().keys())

mydir = "../wideresnet_best.pth.tar"

if os.path.isfile(mydir):
    print("=> loading checkpoint '{}'".format(mydir))
    checkpoint = torch.load(mydir)
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    # print(checkpoint['state_dict'].keys())
    model.load_state_dict(checkpoint['state_dict'])
    # print("=> loaded checkpoint '{}' (epoch {})"
    #         .format(mydir, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(mydir))

# switch to evaluate mode
model.eval()

printparams = {
	'data_root': '../../../data/images/',
    'data_result_list': '../../../data/testprint0.txt',
    'test_num': 10000,
    'model': model,
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean
    }

p = TestPrint(**printparams)
p.PrintToFile()