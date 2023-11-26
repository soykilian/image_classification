import random
import time
import numpy as np
from PIL import Image
from copy import deepcopy
import cpuinfo
import subprocess


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.style.use('fivethirtyeight')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage,\
                                   Resize, CenterCrop, RandomResizedCrop

from torchvision.datasets import ImageFolder, CIFAR10
from torchvision.models   import alexnet, AlexNet_Weights
from torchvision.models   import resnet18, inception_v3
#from torchvision.models.alexnet   import model_urls
#from torchvision.models.hub   import load_state_dict_from_url


from Lec11sbs import StepByStep, CNN2layer
from Lec11ImageStuff import DownloadRPS, plotRPSbatch, plotCIFAR10batch
                           



def NicelyFormatTime(tm):
    if tm<0.001:
        t=tm*1000000.00
        return f"{t:5.1f} \u00b5s"
    elif tm<1.0:             # ms
        t=tm*1000.00
        return f"{t:5.1f} ms"
    elif tm<2.0:
        t=tm*1000.00
        return f"{t:5.0f} ms "
    elif tm<1000.00:
        return f"{tm:5.1f} s "
    elif tm<3600.00:
        return f"{tm:>4.0f} s "
    else:
        t=tm/3600.0
        return f"{t:6.2f} h"



c=cpuinfo.get_cpu_info()
MyCPU=c['brand_raw']
try:
    g=subprocess.check_output('nvidia-smi --query-gpu=gpu_name --format=csv')
    sg=str(g)
    gg=sg.split('\\r\\n')
    MyGPU=gg[1]
except:
    MyGPU="No GPU Available"



def FreezeModel(model):
    # Change all model parameters to "do not learn"
    for par in model.parameters():
        par.requires_grad=False         # Will no longer learn this parameter

       
def PreprocessedDataset(model, loader, device=None):
    if device is None:  
        device = next(model.parameters()).device
        features = None
        labels = None
 
    for i, (x, y) in enumerate(loader):
        model.eval()
        output = model(x.to(device))
        if i == 0:
            features = output.detach().cpu()
            labels = y.cpu()
        else:
            features = torch.cat(
                [features, output.detach().cpu()])
            labels = torch.cat([labels, y.cpu()])

    dataset = TensorDataset(features, labels)
    return dataset




####################################################################
### LOAD THE CIFAR10 DATASET                                     ###
### Each image is 32x32 in this dataset; will resize to 32x32    ###
####################################################################
batch_size=32
CIFAR10NUMTRAIN=50000
CIFAR10NUMTEST=10000
normalizer=Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# Original RPS dataset images are 300x300 (3 channel)
# We need to convert them to 224x224. 

CIFAR10transform    = Compose( [Resize(224), ToTensor(), normalizer ])

CIFAR10trainset     = CIFAR10(root='./data', train=True, download=True, 
                              transform=CIFAR10transform)

CIFAR10trainloader  = DataLoader(CIFAR10trainset, batch_size=batch_size, 
                                 shuffle=True)

CIFAR10testset           = CIFAR10(root='./data', train=False, download=True, 
                                   transform=CIFAR10transform)
CIFAR10testloader   = DataLoader(CIFAR10testset, batch_size=batch_size, 
                                 shuffle=False)

CIFAR10classes      = ('plane', 'car',  'bird',  'cat',  'deer', 
                       'dog',   'frog', 'horse', 'ship', 'truck')

# plotTransform=Compose( [ ToTensor() ])
# plotset = CIFAR10(root='./data', train=False, download=True, 
#                   transform=plotTransform)
# plotloader = DataLoader(plotset, batch_size=60, shuffle=False)

# plotCIFAR10batch(plotloader, 'CIFAR10 Validation batch (3x32x32)', 6, 10, 4)



############################################################
### LOAD THE AlexNet MODEL                               ###
### WE WILL TRAIN IT USING THE CIFAR10 DATASET           ###
############################################################

# Load AlexNet with the weights
ECE655AlexI = alexnet(weights=AlexNet_Weights.DEFAULT)
FreezeModel(ECE655AlexI)
ECE655AlexI.classifier[6]=nn.Identity()     # delete the output layer for preprocessing

###########################################################
###  WE WILL PREPROCESS THE IMAGES USING AlexI         ###
###########################################################
print("Preprocessing CIFAR10 images  ...  ", end='')
t1=time.time()

PPtrain_data = PreprocessedDataset(ECE655AlexI, CIFAR10trainloader)
PPval_data   = PreprocessedDataset(ECE655AlexI, CIFAR10testloader)

torch.save(PPtrain_data.tensors, 'CIFAR10preproc.pth')
torch.save(PPval_data.tensors, 'CIFAR10val_preproc.pth')

# STEP 1: This is where our preprocessed train and val datasets are
x, y = torch.load('CIFAR10preproc.pth')
train_preproc = TensorDataset(x, y)
val_preproc = TensorDataset(*torch.load('CIFAR10val_preproc.pth'))


train_preproc_loader = DataLoader(train_preproc, batch_size=16, shuffle=True)
val_preproc_loader   = DataLoader(val_preproc, batch_size=16)
t2=time.time()

print(f"Done ...  It took {NicelyFormatTime(t2-t1)} to preprocess all images.")




for Epochs in range(1, 30, 3):
    # STEP 2: We will only use the final layer for training
    torch.manual_seed(17)
    ECE655AlexFinalLayer = nn.Sequential(nn.Linear(4096,10))
    LossFN=nn.CrossEntropyLoss(reduction='mean')
    Optim=optim.Adam(ECE655AlexFinalLayer.parameters(), lr=3e-4)
    # Our model to train the final layer
    sbsAlexFinal=StepByStep(ECE655AlexFinalLayer, LossFN,Optim)
    sbsAlexFinal.set_loaders(train_preproc_loader, val_preproc_loader)
    t1=time.time()
    sbsAlexFinal.train(Epochs)
    t2=time.time()
    TT=t2-t1                #Training Time
    TTPE=TT/float(Epochs)   #Training Time per epoch
    TT=NicelyFormatTime(TT)
    TTPE=NicelyFormatTime(TTPE)
    ModelParams=sbsAlexFinal.count_parameters()
    print(f"MODEL NAME: AlexFinal ({ModelParams} parameters)")
    print(f"    CPU:                   {MyCPU}")
    print(f"    GPU:                   {MyGPU}     ",end='')
    if not torch.cuda.is_available():
        print("   !!! NOT USED BY PyTorch !!!")
    else:
        print("   USED BY PyTorch")
    print(f"    TRAINING:             {Epochs} epoch{'s' if Epochs>1 else ''}")
    print(f"    TRAINING TIME:        {TT}  ({TTPE} per epoch)")


    # Step 3: PLUG THIS TRAINED FINAL LAYER BACK INTO THE FINAL LAYER OF ALEX
    ECE655AlexI.classifier[6]=ECE655AlexFinalLayer     # replace the final layer
    # ECE655AlexI is the trained version. We can now make predictions with it
    # Use the "un-preprocessed" data to alidate results
    sbsAlexI=StepByStep(ECE655AlexI, LossFN, Optim)
    
    trainAccuracy=StepByStep.loader_apply(CIFAR10trainloader, sbsAlexI.correct)
    trainPredictions = trainAccuracy[:, 1]
    trainCorrectPred = trainAccuracy[:, 0]
    trainModelAccuracy = trainCorrectPred.sum()/trainPredictions.sum()*100.00
    tpred=trainPredictions.sum()
    tcorrect=trainCorrectPred.sum()
    print(f"   TRAINING   ACCURACY:   {trainModelAccuracy:>5.1f}%",end='')
    print(f"     {tpred:>4} predictions ({tcorrect:>5} correct ones)")

    valAccuracy=StepByStep.loader_apply(CIFAR10testloader, sbsAlexI.correct)
    valPredictions = valAccuracy[:, 1]
    valCorrectPred = valAccuracy[:, 0]
    valModelAccuracy = valCorrectPred.sum()/valPredictions.sum()*100.00
    vpred=valPredictions.sum()
    vcorrect=valCorrectPred.sum()
    print(f"   VALIDATION ACCURACY:   {valModelAccuracy:>5.1f}%",end='')
    print(f"     {vpred:>4} predictions ({vcorrect:>5} correct ones)")
    
    print(80*'=')



