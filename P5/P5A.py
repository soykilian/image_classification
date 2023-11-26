from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageFolder, CIFAR100
import random
import time
import numpy as np
from PIL import Image
from copy import deepcopy
#import cpuinfo
import subprocess


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.style.use('fivethirtyeight')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Lec11ImageStuff import DownloadRPS, plotRPSbatch, plotCIFAR10batch
from Lec11sbs import StepByStep, CNN2layer
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage,\
                                   Resize, CenterCrop, RandomResizedCrop
from dataset import FreezeModel
batch_size=32
CIFAR10NUMTRAIN=50000
CIFAR10NUMTEST=10000
normalizer=Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# Original RPS dataset images are 300x300 (3 channel)
# We need to convert them to 224x224. 

CIFAR100transform    = Compose( [Resize(224), ToTensor(), normalizer ])

CIFAR100trainset     = CIFAR100(root='./data', train=True, download=True, 
                              transform=CIFAR100transform)

CIFAR100trainloader  = DataLoader(CIFAR100trainset, batch_size=batch_size, 
                                 shuffle=True)

CIFAR100testset           = CIFAR100(root='./data', train=False, download=True, 
                                   transform=CIFAR100transform)
CIFAR100testloader   = DataLoader(CIFAR100testset, batch_size=batch_size, 
                                 shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
for name, param in pretrained_resnet.named_parameters():
    if param.requires_grad:
        print(name)

FreezeModel(pretrained_resnet)
pretrained_resnet.fc = nn.Sequential(nn.Linear(2048, 100))
pretrained_resnet.to(device)
for name, param in pretrained_resnet.named_parameters():
    if param.requires_grad:
        print(name)

torch.manual_seed(20)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_resnet.parameters(), lr=3e-4)
sbsResnet = StepByStep(pretrained_resnet, loss_function, optimizer)
sbsResnet.set_loaders(CIFAR100trainloader, CIFAR100testloader)
t1 = time.time()
sbsResnet.train(3)
t2 = time.time()
TT = t2-t1
print(f"Training time: {TT/3:.2f} seconds")
parameters = sbsResnet.count_parameters()
print(f"Total number of parameters: {parameters}")

trainAccuracy=StepByStep.loader_apply(CIFAR100trainloader, sbsResnet.correct)
trainPredictions = trainAccuracy[:, 1]
trainCorrectPred = trainAccuracy[:, 0]
trainModelAccuracy = trainCorrectPred.sum()/trainPredictions.sum()*100.00
tpred=trainPredictions.sum()
tcorrect=trainCorrectPred.sum()
print(f"   TRAINING   ACCURACY:   {trainModelAccuracy:>5.1f}%",end='')
print(f"     {tpred:>4} predictions ({tcorrect:>5} correct ones)")

valAccuracy=StepByStep.loader_apply(CIFAR100testloader, sbsResnet.correct)
valPredictions = valAccuracy[:, 1]
valCorrectPred = valAccuracy[:, 0]
valModelAccuracy = valCorrectPred.sum()/valPredictions.sum()*100.00
vpred=valPredictions.sum()
vcorrect=valCorrectPred.sum()
print(f"   VALIDATION ACCURACY:   {valModelAccuracy:>5.1f}%",end='')
print(f"     {vpred:>4} predictions ({vcorrect:>5} correct ones)")