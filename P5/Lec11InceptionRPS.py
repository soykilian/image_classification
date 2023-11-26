import random
import time
import numpy as np
from PIL import Image
from copy import deepcopy
import cpuinfo
import subprocess


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, random_split

from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage,\
                                   Resize, CenterCrop, RandomResizedCrop

from torchvision.datasets import ImageFolder
from torchvision.models   import alexnet, AlexNet_Weights
from torchvision.models   import inception_v3, Inception_V3_Weights
#from torchvision.models.alexnet   import model_urls
#from torchvision.models.hub   import load_state_dict_from_url


from Lec11sbs import StepByStep, CNN2layer
from Lec11ImageStuff import DownloadRPS, plotRPSbatch
                           


def PrintAccuracy(ModelName, ModelParams, Epochs, valAccuracy, TrainAccuracy, Losses=None):
    NumberOfClasses=valAccuracy.shape[0]

    print(f"MODEL NAME: {ModelName} ({ModelParams} parameters)")
    print(f"   TRAINING:               {Epochs} epoch{'s' if Epochs>1 else ''}")
    if Losses is not None:
        LossT, LossV = Losses
        print(f"   MODEL LOSS:             {LossT:>5.3f} (train)    {LossV:>5.3f} (val) ")

    trainPredictions = TrainAccuracy[:, 1]
    trainCorrectPred = TrainAccuracy[:, 0]
    trainModelAccuracy = trainCorrectPred.sum()/trainPredictions.sum()*100.00
    trainClassAccuracy=trainCorrectPred/trainPredictions*100.00
    tpred=trainPredictions.sum()
    tcorrect=trainCorrectPred.sum()
    print(f"   TRAINING ACCURACY:     {trainModelAccuracy:>5.1f}%",end='')
    print(f"     {tpred:>4} predictions ({tcorrect} correct ones)")
        
    
    for i in range(NumberOfClasses):
        print(f"         Class {i} Accuracy   = {trainClassAccuracy[i]:>5.1f}%", end='')
        print(f" ({trainCorrectPred[i]}/{trainPredictions[i]})")

    valPredictions = valAccuracy[:, 1]
    valCorrectPred = valAccuracy[:, 0]
    valModelAccuracy = valCorrectPred.sum()/valPredictions.sum()*100.00
    valClassAccuracy=valCorrectPred/valPredictions*100.00
    vpred=valPredictions.sum()
    vcorrect=valCorrectPred.sum()
    
    print(f"   VALIDATION ACCURACY:   {valModelAccuracy:>5.1f}%",end='')
    print(f"     {vpred:>4} predictions ({vcorrect} correct ones)")
        
    
    for i in range(NumberOfClasses):
        print(f"         Class {i} Accuracy   = {valClassAccuracy[i]:>5.1f}%", end='')
        print(f" ({valCorrectPred[i]}/{valPredictions[i]})")


# Calculates a loss function, which takes into account auxiliary labels
# Aux losses contribute 40% and the main contributes 60% to the combined loss
def InceptionLoss(outputs, labels):
    try:
        main, aux = outputs
    except ValueError:
        main = outputs
        aux = None
        loss_aux = 0
    
    multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss_main = multi_loss_fn(main, labels)
    if aux is not None:
        loss_aux = multi_loss_fn(aux, labels)
    return loss_main + 0.4 * loss_aux



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


# Load the pretrained Inception model 
ECE655Inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
#print(f"Here is the Inception architecture:\n{ECE655Inception}")
InceptionArchitecture=str(ECE655Inception)
with open('InceptionV3Architecture.txt', 'w') as file:
    file.write(InceptionArchitecture)
print(80*'=')
FreezeModel(ECE655Inception)
# Replace the final layer
torch.manual_seed(42)
ECE655Inception.AuxLogits.fc = nn.Linear(768, 3)
ECE655Inception.fc=nn.Linear(2048,3)



#######################################################
### LOAD THE RPS DATASET                            ###
### For normalization, use pubished mean and std    ###
#######################################################

# For normalization, we will use the published channel statistics;
#    mean=[0.485, 0.456, 0.406]  std=[0.229, 0.224, 0.225]
normalizer=Normalize(mean=[0.485, 0.456, 0.406],  
                      std=[0.229, 0.224, 0.225] )
# Original RPS dataset images are 300x300 (3 channel)
# We need to convert them to 224x224. So, we will CenterCrop
InceptionTransform=Compose([Resize(299), ToTensor(), normalizer])

# Both training and validation data will be handled using ImageFolder
train_data = ImageFolder(root='rps', transform=InceptionTransform)
val_data   = ImageFolder(root='rps-test-set', transform=InceptionTransform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=16)


# Model setup
InceptionOptim = optim.Adam(ECE655Inception.parameters(), lr=3e-4)
sbsInception=StepByStep(ECE655Inception, InceptionLoss, InceptionOptim)
sbsInception.set_loaders(train_loader, val_loader)

ModelParams=sbsInception.count_parameters()


# Train it for 1 epoch
sbsInception.train(1, verbose=True)

# Print accuracy metrics
ValAccuracy=StepByStep.loader_apply(sbsInception.val_loader, sbsInception.correct)
TrainAccuracy=StepByStep.loader_apply(sbsInception.train_loader, sbsInception.correct)
PrintAccuracy('Inception', ModelParams, 1, ValAccuracy, TrainAccuracy)

# Train it for 1 more epoch
sbsInception.train(1, verbose=True)

# Print accuracy metrics
ValAccuracy=StepByStep.loader_apply(sbsInception.val_loader, sbsInception.correct)
TrainAccuracy=StepByStep.loader_apply(sbsInception.train_loader, sbsInception.correct)
PrintAccuracy('Inception', ModelParams, 2, ValAccuracy, TrainAccuracy)


# Train it for 1 more epoch
sbsInception.train(1, verbose=True)

# Print accuracy metrics
ValAccuracy=StepByStep.loader_apply(sbsInception.val_loader, sbsInception.correct)
TrainAccuracy=StepByStep.loader_apply(sbsInception.train_loader, sbsInception.correct)
PrintAccuracy('Inception', ModelParams, 3, ValAccuracy, TrainAccuracy)









