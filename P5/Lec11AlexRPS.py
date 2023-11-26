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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau,\
                                     MultiStepLR, CyclicLR, LambdaLR

from torch.utils.data import DataLoader, Dataset, random_split

from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage,\
                                   Resize, CenterCrop, RandomResizedCrop

from torchvision.datasets import ImageFolder
from torchvision.models   import alexnet, AlexNet_Weights
from torchvision.models   import resnet18, inception_v3
#from torchvision.models.alexnet   import model_urls
#from torchvision.models.hub   import load_state_dict_from_url


from Lec11sbs import StepByStep, CNN2layer
from Lec11ImageStuff import DownloadRPS, plotRPSbatch
                           



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


def PrintAccuracy(ModelName, valAccuracy, TrainAccuracy, Losses=None):
    NumberOfClasses=valAccuracy.shape[0]

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


def AnalyzeAlex(AlexModel, ModelName='Alex', Classifier='012-345-6', Epochs=6, 
                PlotLoss=True, PlotFilt1=False, PlotFilt2=False, NoPlots=True,
                PrintRuntimes=True, PrintConfidences=True):
    
    if NoPlots:
        PlotLoss=False
        PlotFilt1=False
        PlotFilt2=False
    
    torch.manual_seed(13)
    ###    MODEL HYPER-PARAMETERS      
    AlexLoss=nn.CrossEntropyLoss(reduction='mean')
    AlexOptim=optim.Adam(AlexModel.parameters(),lr=3e-4)

    ###    MODEL CONFIGURATION      
    sbsAlex=StepByStep(AlexModel, AlexLoss, AlexOptim)
    sbsAlex.set_loaders(train_loader, val_loader)
    t1=time.time()
    sbsAlex.train(Epochs)
    t2=time.time()
    TT=t2-t1                #Training Time
    TTPE=TT/float(Epochs)   #Training Time per epoch
    TT=NicelyFormatTime(TT)
    TTPE=NicelyFormatTime(TTPE)
    ModelParams=sbsAlex.count_parameters()
    print(f"MODEL NAME: {ModelName} ({ModelParams} parameters)",end='')
    print(f"        CLASSIFIER = {Classifier}")
    print(f"   TRAINING:               {Epochs} epoch{'s' if Epochs>1 else ''}")
    fig=None
    figFILTERS1=None
    figFILTERS2=None
    if PlotLoss:
        fig=sbsAlex.PlotLossesWithInfo(ModelName)
    if PlotFilt1:
        pass
        #figFILTERS1=sbsAlex.visualize_filters('conv1', cmap='gray')
    if PlotFilt2:
        pass
        #figFILTERS2=sbsAlex.visualize_filters('conv2', cmap='gray')
    
    ###  EVALUATE MODEL ACCURACY
    ValAccuracy=StepByStep.loader_apply(sbsAlex.val_loader, sbsAlex.correct)
    TrainAccuracy=StepByStep.loader_apply(sbsAlex.train_loader, sbsAlex.correct)
    LossT=min(sbsAlex.losses)
    LossV=min(sbsAlex.val_losses)
    PrintAccuracy(ModelName, ValAccuracy, TrainAccuracy, (LossT,LossV))
    
    if PrintConfidences:
        VALIMAGES=372
        vloader   = DataLoader(val_data,   batch_size=VALIMAGES)
        images_batch, labels_batch = iter(vloader).__next__()
        t1=time.time()
        logits = sbsAlex.predict(images_batch)
        t2=time.time()
        PredictionTime=(t2-t1)/float(VALIMAGES)*1000.00   # in ms
        predicted = np.argmax(logits, 1)
        ClassConfMax=[0.00, 0.00, 0.00]
        ClassConfMin=[100.00, 100.00, 100.00]
        for LG in range(logits.shape[0]):
            L=torch.tensor(logits[LG])
            P=nn.Softmax(dim=-1)(L)*1000.0
            P=P.cpu().detach().numpy().astype('uint32')
            P=list(P.astype('float32')/10.0)
            CI=predicted[LG]
            prob=P[CI]
            ClassConfMax[CI]=max(ClassConfMax[CI],prob)
            ClassConfMin[CI]=min(ClassConfMin[CI],prob)
        print("   PREDICTION CONFIDENCES:")
        for i in range(3):
            CCmax=ClassConfMax[i]
            CCmin=ClassConfMin[i]
            if CCmin>CCmax:
                print(f"         Class {i} Confidence = (    ? - ?    )")
            else:    
                print(f"         Class {i} Confidence = ({CCmin:4.1f}% - {CCmax:4.1f}%)")

    if PrintRuntimes:
        print(f"   CPU:                {MyCPU}")
        print(f"   GPU:                {MyGPU}     ",end='')
        if not torch.cuda.is_available():
            print("   !!! NOT USED BY PyTorch !!!")
        else:
            print("   USED BY PyTorch")
            
        print(f"   TRAINING TIME:      {TT}  ({TTPE} per epoch)")
        print(f"   EVAL     TIME:      {PredictionTime:6.2f} ms (per image)")

    print(80*'=')                 

    return fig, figFILTERS1, figFILTERS2, ValAccuracy, TrainAccuracy

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
rpsTransform=Compose([Resize(256), CenterCrop(224), ToTensor(), normalizer])

# Both training and validation data will be handled using ImageFolder
train_data = ImageFolder(root='rps', transform=rpsTransform)
val_data   = ImageFolder(root='rps-test-set', transform=rpsTransform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32)





############################################################
### LOAD THE AlexNet MODEL                               ###
### WE WILL USE ITS FEATURES AND MODIFY THE CLASSIFIER   ###
############################################################

# Load AlexNet with the weights
ECE655Alex = alexnet(weights=AlexNet_Weights.DEFAULT)
print("AlexNet Architecture is as follows:")
print(ECE655Alex)
FreezeModel(ECE655Alex)
print("I froze Alex. We will now try different versions applied to the RPS dataset")
print(80*'=')

ECE655Alex.classifier[6]=nn.Linear(4096,3)     # replace the final layer
AnalyzeAlex(ECE655Alex, "Alex012-345-6", Classifier='012-345-6', Epochs=1)
AnalyzeAlex(ECE655Alex, "Alex012-345-6", Classifier='012-345-6', Epochs=2)
AnalyzeAlex(ECE655Alex, "Alex012-345-6", Classifier='012-345-6', Epochs=3)
AnalyzeAlex(ECE655Alex, "Alex012-345-6", Classifier='012-345-6', Epochs=4)
# AnalyzeAlex(ECE655Alex, "Alex012-345-6", Classifier='012-345-6', Epochs=5)
# AnalyzeAlex(ECE655Alex, "Alex012-345-6", Classifier='012-345-6', Epochs=6)
# AnalyzeAlex(ECE655Alex, "Alex012-345-6", Classifier='012-345-6', Epochs=8)
# AnalyzeAlex(ECE655Alex, "Alex012-345-6", Classifier='012-345-6', Epochs=10, NoPlots=False)


ECE655Alex.classifier[3]=nn.Identity()     # remove the second hidden layer
ECE655Alex.classifier[4]=nn.Identity()
ECE655Alex.classifier[5]=nn.Identity()
ECE655Alex.classifier[6]=nn.Linear(4096,3)     # replace the final layer
AnalyzeAlex(ECE655Alex, "Alex012--6", Classifier='012--6', Epochs=1)
AnalyzeAlex(ECE655Alex, "Alex012--6", Classifier='012--6', Epochs=2)
AnalyzeAlex(ECE655Alex, "Alex012--6", Classifier='012--6', Epochs=3)
# AnalyzeAlex(ECE655Alex, "Alex012--6", Classifier='012--6', Epochs=5)
# AnalyzeAlex(ECE655Alex, "Alex012--6", Classifier='012--6', Epochs=10, NoPlots=False)


ECE655Alex.classifier[0]=nn.Identity()     # remove the entire classifier!
ECE655Alex.classifier[1]=nn.Identity()
ECE655Alex.classifier[2]=nn.Identity()
ECE655Alex.classifier[6]=nn.Linear(9216,3)     # replace the final layer
AnalyzeAlex(ECE655Alex, "Alex---6", Classifier='---6', Epochs=1)
AnalyzeAlex(ECE655Alex, "Alex---6", Classifier='---6', Epochs=2)
AnalyzeAlex(ECE655Alex, "Alex---6", Classifier='---6', Epochs=3)
# AnalyzeAlex(ECE655Alex, "Alex---6", Classifier='---6', Epochs=5)
# AnalyzeAlex(ECE655Alex, "Alex---6", Classifier='---6', Epochs=10, NoPlots=False)















