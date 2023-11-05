import random
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.style.use('fivethirtyeight')

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, random_split, \
                             WeightedRandomSampler, SubsetRandomSampler

from torchvision.transforms import Compose, ToTensor, Normalize, \
                                   ToPILImage, RandomHorizontalFlip, \
                                   RandomVerticalFlip, Resize

from Lec8sbs import StepByStep
from Lec8ImageStuff import GenerateImages, PlotImages, PlotPILimg, \
                           PlotPILflipped, PlotAllActivationFunctions
                           



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
        img_size=10, n_images=1000, binary=False, seed=17)

print("Generate Images done")
# Display the first 60 of the 300 images
PlotImages(Images, Labels, nPlot=60, ImgPerRow=10)



##############################################################
######     DATA PREPARATION AND FEATURE NORMALIZATION   ######
######     INCORPORATES DATA AUGMENTATION               ######
##############################################################
# Builds tensors from Numpy arrays BEFORE  split
x_tensor = torch.as_tensor(Images/255.0).float()
y_tensor = torch.as_tensor(Labels).long()

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
print(y_val_tensor)

train_composer = Compose([Normalize(mean=(.5,), std=(.5,))])
# No augmentation for the validation  dataset
val_composer = Compose([Normalize(mean=(.5,), std=(.5,))])

train_dataset = TransformedTensorDataset(
    x_train_tensor, y_train_tensor, transform=train_composer)

val_dataset = TransformedTensorDataset(
    x_val_tensor, y_val_tensor, transform=val_composer)

# Builds a weighted random sampler to handle imbalanced classes
sampler = make_balanced_sampler(y_train_tensor)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=16, sampler=sampler)

val_loader = DataLoader(dataset=val_dataset, batch_size=16)


########################################################################
######     ECE655MCICmodel1 MODEL CONFIGURATION - FEATURIZER     #######
######     1@10x10 -> 1F@8x8 -> 1F@4x4 -> 1F@16 -> 1F@10 -> C=3  #######  
########################################################################
torch.manual_seed(13)
ECE655MCICmodel1 = nn.Sequential()
# Block 1: 1@10x10 -> Features@8x8 -> Features@4x4
Features = 1
ECE655MCICmodel1.add_module('conv1', nn.Conv2d(in_channels=1, 
                                               out_channels=Features, kernel_size=3))
ECE655MCICmodel1.add_module('relu1', nn.ReLU())
ECE655MCICmodel1.add_module('maxp1', nn.MaxPool2d(kernel_size=2))
# Flattening: Features@4x4 -> Features@16
ECE655MCICmodel1.add_module('flatten', nn.Flatten())


########################################################################
######     ECE655MCICmodel1 MODEL CONFIGURATION - CLASSIFIER     #######
########################################################################
# Hidden Layer
ECE655MCICmodel1.add_module('fc1', nn.Linear(in_features=Features*4*4, out_features=10))
ECE655MCICmodel1.add_module('relu2', nn.ReLU())
# Output Layer
ECE655MCICmodel1.add_module('fc2', nn.Linear(in_features=10, out_features=3))


########################################################################
########      MODEL CONFIGURATION - HYPER-PARAMETERS       #############
########################################################################
lr = 0.1
MCIClossfn1 = nn.CrossEntropyLoss(reduction='mean')
MCICoptimizer1 = optim.SGD(ECE655MCICmodel1.parameters(), lr=lr)

sbsMCIC1=StepByStep(ECE655MCICmodel1, MCIClossfn1, MCICoptimizer1)
sbsMCIC1.set_loaders(train_loader, val_loader)
sbsMCIC1.train(24)
fig=sbsMCIC1.PlotLossesWithInfo('MCICmodel1')

figFILTERS=sbsMCIC1.visualize_filters('conv1', cmap='gray')



########################################################################
######     ECE655MCICmodel2      HAS 2 FEATURES INSTEAD OF 1     #######
######     1@10x10 -> 2F@8x8 -> 2F@4x4 -> 1F@32 -> 1F@10 -> C=3  #######  
########################################################################
torch.manual_seed(13)
ECE655MCICmodel2 = nn.Sequential()
# Block 1: 1@10x10 -> Features@8x8 -> Features@4x4
Features = 2
ECE655MCICmodel2.add_module('conv1', nn.Conv2d(in_channels=1, 
                                               out_channels=Features, kernel_size=3))
ECE655MCICmodel2.add_module('relu1', nn.ReLU())
ECE655MCICmodel2.add_module('maxp1', nn.MaxPool2d(kernel_size=2))
# Flattening: Features@4x4 -> Features@32
ECE655MCICmodel2.add_module('flatten', nn.Flatten())

# Hidden Layer
ECE655MCICmodel2.add_module('fc1', nn.Linear(in_features=Features*4*4, out_features=10))
ECE655MCICmodel2.add_module('relu2', nn.ReLU())
# Output Layer
ECE655MCICmodel2.add_module('fc2', nn.Linear(in_features=10, out_features=3))

lr = 0.1
MCIClossfn2 = nn.CrossEntropyLoss(reduction='mean')
MCICoptimizer2 = optim.SGD(ECE655MCICmodel2.parameters(), lr=lr)

sbsMCIC2=StepByStep(ECE655MCICmodel2, MCIClossfn2, MCICoptimizer2)
sbsMCIC2.set_loaders(train_loader, val_loader)
sbsMCIC2.train(24)
fig=sbsMCIC2.PlotLossesWithInfo('MCICmodel2')

figFILTERS=sbsMCIC2.visualize_filters('conv1', cmap='gray')


########################################################################
######     ECE655MCICmodel3 HAS 1 FEATURE and  a smaller classifier ####
######     1@10x10 -> 1F@8x8 -> 1F@4x4 -> 1F@16 -> 1F@5 -> C=3   #######  
########################################################################
torch.manual_seed(13)
ECE655MCICmodel3 = nn.Sequential()
# Block 1: 1@10x10 -> Features@8x8 -> Features@4x4
Features = 1
ECE655MCICmodel3.add_module('conv1', nn.Conv2d(in_channels=1, 
                                               out_channels=Features, kernel_size=3))
ECE655MCICmodel3.add_module('relu1', nn.ReLU())
ECE655MCICmodel3.add_module('maxp1', nn.MaxPool2d(kernel_size=2))
# Flattening: Features@4x4 -> Features@16
ECE655MCICmodel3.add_module('flatten', nn.Flatten())

# Hidden Layer
ECE655MCICmodel3.add_module('fc1', nn.Linear(in_features=Features*4*4, out_features=5))
ECE655MCICmodel3.add_module('relu2', nn.ReLU())
# Output Layer
ECE655MCICmodel3.add_module('fc2', nn.Linear(in_features=5, out_features=3))

lr = 0.1
MCIClossfn3 = nn.CrossEntropyLoss(reduction='mean')
MCICoptimizer3 = optim.SGD(ECE655MCICmodel3.parameters(), lr=lr)

sbsMCIC3=StepByStep(ECE655MCICmodel3, MCIClossfn3, MCICoptimizer3)
sbsMCIC3.set_loaders(train_loader, val_loader)
sbsMCIC3.train(24)
fig=sbsMCIC3.PlotLossesWithInfo('MCICmodel3')

figFILTERS=sbsMCIC3.visualize_filters('conv1', cmap='gray')

