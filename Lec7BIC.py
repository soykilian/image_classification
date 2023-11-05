import random
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.style.use('fivethirtyeight')

import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F

from torch.utils.data import DataLoader, Dataset, random_split, \
                             WeightedRandomSampler, SubsetRandomSampler

from torchvision.transforms import Compose, ToTensor, Normalize, \
                                   ToPILImage, RandomHorizontalFlip, \
                                   RandomVerticalFlip, Resize

from Lec7sbs import StepByStep
from Lec7ImageStuff import GenerateImages, PlotImages, PlotPILimg



################################################################
######    OUR DATASET CAPABLE OF HANDLING TRANSFORMATIONS ######
################################################################
class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
    
    def __getitem__(self, index):
        x = self.x[index]
 
        if self.transform:
            x = self.transform(x)
 
        return x, self.y[index]

    def __len__(self):
        return len(self.x)


#######################################################
########       DATA GENERATION            #############
#######################################################
Images, Labels = GenerateImages(
        img_size=5, n_images=300, binary=True, seed=13)

print("Generate Images done")
# Display the first 60 of the 300 images
PlotImages(Images, Labels, nPlot=60, ImgPerRow=10)



##############################################################
######     DATA PREPARATION AND FEATURE NORMALIZATION   ######
######     INCORPORATES DATA AUGMENTATION               ######
##############################################################
# Builds tensors from Numpy arrays BEFORE  split
x_tensor = torch.as_tensor(Images/255.0).float()
y_tensor = torch.as_tensor(Labels.reshape(-1, 1)).float()

######     SPLIT DATASET INTO TRAINING, VALIDATION      ######

def index_splitter(n, splits, seed=13):
    idx = torch.arange(n)
    # Makes the split argument a tensor
    splits_tensor = torch.as_tensor(splits)
    # Finds the correct multiplier, so we don't have
    # to worry about summing up to N (or one)
    multiplier = n / splits_tensor.sum()
    splits_tensor = (multiplier * splits_tensor).long()
    # If there is a difference, throws at the first split
    # so random_split does not complain
    diff = n - splits_tensor.sum()
    splits_tensor[0] += diff
    # Uses PyTorch random_split to split the indices
    torch.manual_seed(seed)
    return random_split(idx, splits_tensor)

def make_balanced_sampler(y):
    # Computes weights for compensating imbalanced classes
    classes, counts = y.unique(return_counts=True)
    weights = 1.0 / counts.float()
    sample_weights = weights[y.squeeze().long()]
    # Builds sampler with compute weights
    generator = torch.Generator()
    sampler = WeightedRandomSampler(
    weights=sample_weights,
            num_samples=len(sample_weights),
            generator=generator,
            replacement=True)
    return sampler


train_idx, val_idx = index_splitter(len(x_tensor), [80,20])

# Uses indices to perform the split
x_train_tensor = x_tensor[train_idx]
y_train_tensor = y_tensor[train_idx]
x_val_tensor = x_tensor[val_idx]
y_val_tensor = y_tensor[val_idx]

train_composer = Compose([RandomHorizontalFlip(p=.5),
                          Normalize(mean=(.5,), std=(.5,))])
# No augmentation for the validation  dataset
val_composer = Compose([Normalize(mean=(.5,), std=(.5,))])

train_dataset = TransformedTensorDataset(
    x_train_tensor, y_train_tensor, transform=train_composer)

val_dataset = TransformedTensorDataset(
    x_val_tensor, y_val_tensor, transform=val_composer)

# We can use shuffle if we are not using a sampler
sampler = make_balanced_sampler(y_train_tensor)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=16, sampler=sampler)

val_loader = DataLoader(dataset=val_dataset, batch_size=16)



#####################################################################
########     MODEL CONFIGURATION    ECE655model1        #############
#####################################################################
lr = 0.1

torch.manual_seed(17)
ECE655BICmodel1 = nn.Sequential()
ECE655BICmodel1.add_module('flatten', nn.Flatten())
ECE655BICmodel1.add_module('linear', nn.Linear(25, 1, bias=False))
ECE655BICmodel1.add_module('sigmoid', nn.Sigmoid())

# Defines an SGD optimizer to update the parameters
optimizer = optim.SGD(ECE655BICmodel1.parameters(), lr=lr)
# Defines a BCE without logits loss function (uses probabilities)
loss_fn = nn.BCELoss()

# MODEL TRAINING
n_epochs = 100

sbsBIC1=StepByStep(ECE655BICmodel1, loss_fn, optimizer)
sbsBIC1.set_loaders(train_loader, val_loader)
sbsBIC1.train(n_epochs)
plt.figure()
fig=sbsBIC1.PlotLossesWithInfo('ECE655model1')
plt.savefig('./ECE655model1.png')



#####################################################################
######     MODEL CONFIG/TRAIN   ECE655model1b  (with bias)    #######
#####################################################################
ECE655BICmodel1b = nn.Sequential()
ECE655BICmodel1b.add_module('flatten', nn.Flatten())
ECE655BICmodel1b.add_module('linear', nn.Linear(25, 1, bias=True))
ECE655BICmodel1b.add_module('sigmoid', nn.Sigmoid())
optimizer = optim.SGD(ECE655BICmodel1b.parameters(), lr=lr)

sbsBIC1b=StepByStep(ECE655BICmodel1b, loss_fn, optimizer)
sbsBIC1b.set_loaders(train_loader, val_loader)
sbsBIC1b.train(n_epochs)

plt.figure()
fig=sbsBIC1b.PlotLossesWithInfo('ECE655model1b')
plt.savefig('./ECE655model1b.png')



#####################################################################
######     MODEL CONFIG/TRAIN   ECE655model2      no bias     #######
#####################################################################
ECE655BICmodel2 = nn.Sequential()
ECE655BICmodel2.add_module('flatten', nn.Flatten())
ECE655BICmodel2.add_module('hidden0', nn.Linear(25, 5, bias=False))
ECE655BICmodel2.add_module('hidden1', nn.Linear(5,  3, bias=False))
ECE655BICmodel2.add_module('output',  nn.Linear(3,  1, bias=False))
ECE655BICmodel2.add_module('sigmoid', nn.Sigmoid())
optimizer = optim.SGD(ECE655BICmodel2.parameters(), lr=lr)

sbsBIC2=StepByStep(ECE655BICmodel2, loss_fn, optimizer)
sbsBIC2.set_loaders(train_loader, val_loader)
sbsBIC2.train(n_epochs)

fig=sbsBIC2.PlotLossesWithInfo('ECE655model2')


#####################################################################
######     MODEL CONFIG/TRAIN   ECE655model2b    with bias    #######
#####################################################################
ECE655BICmodel2b = nn.Sequential()
ECE655BICmodel2b.add_module('flatten', nn.Flatten())
ECE655BICmodel2b.add_module('hidden0', nn.Linear(25, 5, bias=True))
ECE655BICmodel2b.add_module('hidden1', nn.Linear(5,  3, bias=True))
ECE655BICmodel2b.add_module('output',  nn.Linear(3,  1, bias=True))
ECE655BICmodel2b.add_module('sigmoid', nn.Sigmoid())
optimizer = optim.SGD(ECE655BICmodel2b.parameters(), lr=lr)

sbsBIC2b=StepByStep(ECE655BICmodel2b, loss_fn, optimizer)
sbsBIC2b.set_loaders(train_loader, val_loader)
sbsBIC2b.train(n_epochs)

fig=sbsBIC2b.PlotLossesWithInfo('ECE655model2b')


#####################################################################
######     MODEL CONFIG/TRAIN   ECE655model3                  #######
#####################################################################
ECE655BICmodel3 = nn.Sequential()
ECE655BICmodel3.add_module('flatten', nn.Flatten())
ECE655BICmodel3.add_module('hidden0', nn.Linear(25, 5, bias=True))
ECE655BICmodel3.add_module('activation0', nn.ReLU())
ECE655BICmodel3.add_module('hidden1', nn.Linear(5,  3, bias=True))
ECE655BICmodel3.add_module('activation1', nn.ReLU())
ECE655BICmodel3.add_module('output',  nn.Linear(3,  1, bias=True))
ECE655BICmodel3.add_module('sigmoid', nn.Sigmoid())
optimizer = optim.SGD(ECE655BICmodel3.parameters(), lr=lr)

sbsBIC3=StepByStep(ECE655BICmodel3, loss_fn, optimizer)
sbsBIC3.set_loaders(train_loader, val_loader)
sbsBIC3.train(n_epochs)

fig=sbsBIC3.PlotLossesWithInfo('ECE655model3')

