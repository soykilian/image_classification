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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Lec11ImageStuff import DownloadRPS, plotRPSbatch, plotCIFAR10batch
from Lec11sbs import StepByStep, CNN2layer
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage,\
                                   Resize, CenterCrop, RandomResizedCrop

from model_utils import FreezeModel, PreprocessedDataset
batch_size=128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PB_Resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
FreezeModel(PB_Resnet)
PB_Resnet.fc = nn.Identity()
PB_Resnet.to(device)

x, y = torch.load('CIFAR100preproc.pth')
train_preproc = TensorDataset(x, y)
val_preproc = TensorDataset(*torch.load('CIFAR100val_preproc.pth'))
train_preproc_loader = DataLoader(train_preproc, batch_size=128, shuffle=True)
val_preproc_loader   = DataLoader(val_preproc, batch_size=128)


top_one_train = []
top_one_test = []
top_five_test = []
class_head = nn.Sequential(nn.Linear(2048,100))
class_head.to(device)
torch.manual_seed(17)
loss_fn=nn.CrossEntropyLoss(reduction='mean')
optimizer=optim.Adam(class_head.parameters(), lr=3e-4)
# Our model to train the final layer
sbs_resnet_final=StepByStep(class_head, loss_fn,optimizer)
sbs_resnet_final.set_loaders(train_preproc_loader, val_preproc_loader)
sbs_resnet_final.to(device)
for Epochs in range(1,30):
    t1=time.time()
    sbs_resnet_final.train(Epochs)
    t2=time.time()
    TT=t2-t1                #Training Time
    TTPE=TT/float(Epochs)   #Training Time per epoch
    TT=NicelyFormatTime(TT)
    TTPE=NicelyFormatTime(TTPE)
    params=sbs_resnet_final.count_parameters()
    print(f"    TRAINING:             {Epochs} epoch{'s' if Epochs>1 else ''}")
    print(f"    TRAINING TIME:        {TT}  ({TTPE} per epoch)")
    trainAccuracy=StepByStep.loader_apply(train_preproc_loader, sbs_resnet_final.correct)
    trainPredictions = trainAccuracy[:, 1]
    trainCorrectPred = trainAccuracy[:, 0]
    trainModelAccuracy = trainCorrectPred.sum()/trainPredictions.sum()*100.00
    top_one_train.append(trainModelAccuracy)
    tpred=trainPredictions.sum()
    tcorrect=trainCorrectPred.sum()
    print(f"   TRAINING   ACCURACY:   {trainModelAccuracy:>5.1f}%",end='')
    print(f"     {tpred:>4} predictions ({tcorrect:>5} correct ones)")
    valAccuracy, top_5_val=StepByStep.loader_apply(val_preproc_loader, sbs_resnet_final.correct, top5=True)
    valPredictions = valAccuracy[:, 1]
    valCorrectPred = valAccuracy[:, 0]
    valModelAccuracy = valCorrectPred.sum()/valPredictions.sum()*100.00
    top_five_test.append(top_5_val)
    top_one_test.append(valModelAccuracy)
    vpred=valPredictions.sum()
    vcorrect=valCorrectPred.sum()
    print(f"   VALIDATION ACCURACY:   {valModelAccuracy:>5.1f}%",end='')
    print(f" TOP-5: {top_5_val:>5.1f}%")
    print(f"     {vpred:>4} predictions ({vcorrect:>5} correct ones)")
    print(80*'=')


plt.figure()
plt.plot(range(1,30),top_one_train, label='Training Top-1 accuracy', color='blue')
plt.plot(range(1,30),top_one_test, label='Test Top-1 accuracy', color='red')
plt.plot(range(1,30),top_five_test, label='Test Top-5 accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('PB.png')