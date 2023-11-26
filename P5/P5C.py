from torchvision.models import resnet50, ResNet50_Weights
from dataset import NicelyFormatTime
from torchvision.datasets import ImageFolder, CIFAR100
import random
import time
import numpy as np
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.style.use('fivethirtyeight')
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Lec11ImageStuff import DownloadRPS, plotRPSbatch, plotCIFAR10batch
from Lec11sbs import StepByStep, CNN2layer
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage,\
                                   Resize, CenterCrop, RandomResizedCrop
from model_utils import FreezeModel, PreprocessedDatasetC
"""
resnetPC = resnet50(weights=ResNet50_Weights.DEFAULT)
FreezeModel(resnetPC)
resnetPC.fc = nn.Identity()
PPtrain_data = PreprocessedDatasetC(resnetPC, CIFAR100trainloader)
PPval_data   = PreprocessedDatasetC(resnetPC, CIFAR100testloader)
torch.save(PPtrain_data.tensors, 'CIFAR100preproc_C.pth')
torch.save(PPval_data.tensors, 'CIFAR100val_preproc_C.pth')
"""
x, y = torch.load('CIFAR100preproc_C.pth')
train_preproc = TensorDataset(x, y)
val_preproc = TensorDataset(*torch.load('CIFAR100val_preproc_C.pth'))
# 20 final classes 
train_preproc_loader_20 = DataLoader(train_preproc, batch_size=128, shuffle=True)
val_preproc_loader_20   = DataLoader(val_preproc, batch_size=128)

SuperFinal = nn.Sequential(nn.Linear(2048, 20))
loss1 = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(SuperFinal.parameters(), lr=3e-4)
sbsSuperFinal = StepByStep(SuperFinal, loss1, optimizer1)
sbsSuperFinal.set_loaders(train_preproc_loader_20, val_preproc_loader_20)
train_acc = []
val_acc = []

for epoch in tqdm(range(30)):
    sbsSuperFinal.train(epoch)
    trainAccuracy = StepByStep.loader_apply(train_preproc_loader_20, sbsSuperFinal.correct)
    trainPredictions = trainAccuracy[:, 1]
    trainCorrectPred = trainAccuracy[:, 0]
    trainModelAccuracy = trainCorrectPred.sum()/trainPredictions.sum()*100.00
    train_acc.append(trainModelAccuracy)
    valAccuracy = StepByStep.loader_apply(val_preproc_loader_20, sbsSuperFinal.correct)
    valPredictions = valAccuracy[:, 1]
    valCorrectPred = valAccuracy[:, 0]
    valModelAccuracy = valCorrectPred.sum()/valPredictions.sum()*100.00
    val_acc.append(valModelAccuracy)
    print(f"Epoch: {epoch+1} Train Accuracy: {trainModelAccuracy:.2f} Val Accuracy: {valModelAccuracy:.2f}")


x, y = torch.load('CIFAR100preproc.pth')
train_preproc = TensorDataset(x, y)
val_preproc = TensorDataset(*torch.load('CIFAR100val_preproc.pth'))

train_preproc_loader = DataLoader(train_preproc, batch_size=128, shuffle=True)
val_preproc_loader   = DataLoader(val_preproc, batch_size=128)

SubFinal = nn.Sequential(nn.Linear(2048, 100))
loss2 = nn.CrossEntropyLoss() 
optimizer2 = optim.Adam(SubFinal.parameters(), lr=3e-4)
sbsSubFinal = StepByStep(SubFinal, loss2, optimizer2)
sbsSubFinal.set_loaders(train_preproc_loader, val_preproc_loader)
train_acc_2 = []
val_acc_2 = []
for epoch in tqdm(range(30)):
    sbsSubFinal.train(epoch)
    trainAccuracy = StepByStep.loader_apply(train_preproc_loader, sbsSubFinal.correct)
    trainPredictions = trainAccuracy[:, 1]
    trainCorrectPred = trainAccuracy[:, 0]
    trainModelAccuracy = trainCorrectPred.sum()/trainPredictions.sum()*100.00
    train_acc_2.append(trainModelAccuracy)
    valAccuracy = StepByStep.loader_apply(val_preproc_loader, sbsSubFinal.correct)
    valPredictions = valAccuracy[:, 1]
    valCorrectPred = valAccuracy[:, 0]
    valModelAccuracy = valCorrectPred.sum()/valPredictions.sum()*100.00
    val_acc_2.append(valModelAccuracy)
    print(f"Epoch: {epoch+1} Train Accuracy: {trainModelAccuracy:.2f} Val Accuracy: {valModelAccuracy:.2f}")

plt.figure()
plt.plot(range(1,31),train_acc, label='SuperFinal: training accuracy', marker='o')
plt.plot(range(1,31),val_acc, label='SuperFinal: validation accuracy', marker='o')
plt.plot(range(1,31),train_acc_2, label='SubFinal: training accuracy', marker='o')
plt.plot(range(1,31),val_acc_2, label='SubFinal: validation accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.savefig('SuperFinal_SubFinal.png')

torch.save(SuperFinal.state_dict(), 'SuperFinal.pth')
torch.save(SubFinal.state_dict(), 'SubFinal.pth')