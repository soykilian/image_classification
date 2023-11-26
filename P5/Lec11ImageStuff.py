############################################################################
###   LECTURE 9 IMAGE DATA GENERATION AND PLOTTING MANIPULATION CODE     ###
###   ADAPTED, MODIFIED BY DR. TOLGA SOYATA     10/21/2023               ###
###   FROM SOURCE: DANIEL VOIGT GODOY                                    ###
############################################################################

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import torch
import torch.nn.functional as F

from torchvision.transforms import Compose, ToTensor, Normalize, \
                                   ToPILImage, RandomHorizontalFlip, \
                                   RandomVerticalFlip, Resize


    
def PlotImages(images, targets, nPlot=30, ImgPerRow=6):
    nRows = nPlot // ImgPerRow + ((nPlot % ImgPerRow) > 0)
    fig, axes = plt.subplots(nRows, ImgPerRow, 
                             figsize=(ImgPerRow*1.4, 1.6 * nRows))
    axes = np.atleast_2d(axes)
    
    for i, (image, target) in enumerate(zip(images[:nPlot], targets[:nPlot])):
        row, col = i // ImgPerRow, i % ImgPerRow    
        ax = axes[row, col]
        ax.set_title('#{}: Label={}'.format(i, target), {'size': 12})
        # plot filter channel in grayscale
        #ax.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.imshow(image.squeeze(), vmin=0, vmax=1)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.label_outer()

    plt.tight_layout()
    return fig


    
def PlotPILimg(PILimg):
    plt.imshow(PILimg, cmap='gray')
    plt.grid(False)
    
    
def PlotTensor(ImgTensor, ImgTitle):
    plt.figure()
    if ImgTensor.ndim == 4:
        T=torch.squeeze(ImgTensor, axis=0)
    elif ImgTensor.ndim == 2:
        T=torch.unsqueeze(ImgTensor, axis=0)
    else:
        T=ImgTensor
        
    ImgPIL=ToPILImage()(T)
    if T.size()[0]==1:
        plt.imshow(ImgPIL, cmap='gray')
    else:
        plt.imshow(ImgPIL)
    plt.title(ImgTitle)
    #plt.grid(False)

    
# plots the original, horizontal flipped, and vertical flipped 
# versions of a PIL image (in CHW format)
def PlotPILflipped(ImgCHW):
    ImgHWC=np.transpose(ImgCHW, (1,2,0))
    tensorizer=ToTensor()
    ImgTensor=tensorizer(ImgHWC)
    ImgPIL=ToPILImage()(ImgTensor)
    flipperH=RandomHorizontalFlip(p=1.0)
    flipperV=RandomVerticalFlip(p=1.0)
    ImgHflip=flipperH(ImgPIL)
    ImgVflip=flipperV(ImgPIL)
    PlotPILimg(ImgPIL)
    PlotPILimg(ImgHflip)    
    PlotPILimg(ImgVflip)
    
    
    
def PlotActivationFunction(func, name, yLabel):
    z = torch.linspace(-5, 5, 1000)
    z.requires_grad_(True)
    func(z).sum().backward()
    sig = func(z).detach()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    if name is None:
        try:
            name = func.__name__
        except AttributeError:
            name = ''

    if name == 'sigmoid':
        ax.set_ylim([0, 1.1])
    elif name == 'tanh':
        ax.set_ylim([-1.1, 1.1])
    elif name == 'ReLU':
        ax.set_ylim([-.1, 5.01])
    else:
        ax.set_ylim([-1.1, 5.01])
        
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_xlabel('z')
    ax.set_ylabel(yLabel)

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_title(name, fontsize=16)
    ax.plot(z.detach().numpy(), sig.numpy(), c='k', label='Activation')
    ax.plot(z.detach().numpy(), z.grad.numpy(), c='r', label='Gradient')
    ax.legend(loc=2)

    fig.tight_layout()
    fig.show()
    return fig


def PlotAllActivationFunctions():
    PlotActivationFunction(torch.sigmoid,'sigmoid', r'$\sigma(z)$')
    PlotActivationFunction(torch.tanh,'tanh', r'$\tanh(z)$')
    PlotActivationFunction(torch.relu,'ReLU', r'$ReLU(z)$')
    PlotActivationFunction(F.leaky_relu,'Leaky ReLU 0.01', r'$Leaky ReLU(z)$')
    
    
    
####################################################################
###   THIS FUNCTION DOWNLOADS RPS DATABASE (ROCK SCISSORS PAPER)
####################################################################
import requests
import zipfile
import os
import errno

def DownloadRPS(localfolder=''):
    filenames = ['rps.zip', 'rps-test-set.zip']
    for filename in filenames:
        try:
            os.mkdir(f'{localfolder}{filename[:-4]}')

            localfile = f'{localfolder}{filename}'
            # url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/{}'
            # Updated from TFDS URL at
            # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/rock_paper_scissors/rock_paper_scissors_dataset_builder.py
            url = 'https://storage.googleapis.com/download.tensorflow.org/data/{}'
            r = requests.get(url.format(filename), allow_redirects=True)
            open(localfile, 'wb').write(r.content)
            with zipfile.ZipFile(localfile, 'r') as zip_ref:
                zip_ref.extractall(localfolder)        
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            else:
                print(f'{filename[:-4]} folder already exists!')


def plotRPSbatch(Loader, suptitle):
    plt.figure()            
    images, labels = next(iter(Loader))
    print(f"Plotting  {suptitle}")
    print(f"    Loader={Loader}\n    images.size = {images.size()}")
    imgPerRow=8
    fig, axs = plt.subplots(2, imgPerRow, figsize=(12, 4))
    fig.suptitle(suptitle)
    titles = ['Paper', 'Rock', 'Scissors']
    for j in range(2):
        for i in range(imgPerRow):
            idx=i+j*imgPerRow
            image, label = ToPILImage()(images[idx]), labels[idx]
            axs[j][i].imshow(image)
            axs[j][i].set_xticks([])
            axs[j][i].set_yticks([])
            PlotTitle=f"{idx}: {titles[label]}"
            axs[j][i].set_title(PlotTitle, fontsize=12)
    fig.tight_layout()
    return fig

 
def plotCIFAR10batch(Loader, suptitle, plotRows, plotCols, numPlots=1):
    LoaderIter=iter(Loader)
    for x in range(numPlots):
        TopTitle = suptitle + f" (Batch {x})"
        plt.figure()            
        images, labels = next(LoaderIter)
        #print(f"Plotting  {TopTitle}")
        #print(f"    Loader={Loader}\n    images.size = {images.size()}")
        #imgPerRow=8
        fig, axs = plt.subplots(plotRows, plotCols, figsize=(12, 8))
        fig.suptitle(TopTitle)
        titles = ['plane', 'car',  'bird',  'cat',  'deer', 
                  'dog',   'frog', 'horse', 'ship', 'truck']
        for j in range(plotRows):
            for i in range(plotCols):
                idx=i+j*plotCols
                image, label = ToPILImage()(images[idx]), labels[idx]
                axs[j][i].imshow(image)
                axs[j][i].set_xticks([])
                axs[j][i].set_yticks([])
                PlotTitle=f"{idx}: {titles[label]}"
                axs[j][i].set_title(PlotTitle, fontsize=10)
    fig.tight_layout()
    return fig

 