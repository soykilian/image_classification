############################################################################
###   LECTURE 7 IMAGE DATA GENERATION AND PLOTTING MANIPULATION CODE     ###
###   ADAPTED, MODIFIED BY DR. TOLGA SOYATA     9/29/2023                ###
###   FROM SOURCE: DANIEL GODOY                                          ###
############################################################################

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from torchvision.transforms import Compose, ToTensor, Normalize, \
                                   ToPILImage, RandomHorizontalFlip, \
                                   RandomVerticalFlip, Resize



def gen_img(start, target, fill=1, img_size=10):
    # Generates empty image
    img = np.zeros((img_size, img_size), dtype=np.float64)

    start_row, start_col = None, None

    if start > 0:
        start_row = start
    else:
        start_col = np.abs(start)

    if target == 0:
        if start_row is None:
            img[:, start_col] = fill
        else:
            img[start_row, :] = fill
    else:
        if start_col == 0:
            start_col = 1
        
        if target == 1:
            if start_row is not None:
                up = (range(start_row, -1, -1), 
                      range(0, start_row + 1))
            else:
                up = (range(img_size - 1, start_col - 1, -1), 
                      range(start_col, img_size))
            img[up] = fill
        else:
            if start_row is not None:
                down = (range(start_row, img_size, 1), 
                        range(0, img_size - start_row))
            else:
                down = (range(0, img_size - 1 - start_col + 1), 
                        range(start_col, img_size))
            img[down] = fill
    
    return 255 * img.reshape(1, img_size, img_size)



def GenerateImages(img_size=10, n_images=100, binary=True, seed=17):
    np.random.seed(seed)

    starts = np.random.randint(-(img_size - 1), img_size, size=(n_images,))
    targets = np.random.randint(0, 3, size=(n_images,))
    
    images = np.array([gen_img(s, t, img_size=img_size) 
                       for s, t in zip(starts, targets)], dtype=np.uint8)
    
    if binary:
        targets = (targets > 0).astype(np.uint8)

    return images, targets
    

    
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
        ax.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.label_outer()

    plt.tight_layout()
    return fig


    
def PlotPILimg(PILimg):
    plt.imshow(PILimg, cmap='gray')
    plt.grid(False)
    
    
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
    