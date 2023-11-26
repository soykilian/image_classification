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
from CustomDataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model_utils import BrainNet


transform = transforms.Compose([
    transforms.Resize((32, 32)),         # Resize to 32x32 pixels
    transforms.ToTensor()                # Convert to tensor
])

# Create an instance of your custom dataset
root_dir = './NEW_CIFAR/'
custom_dataset = CustomDataset(root_dir, transform=transform)
batch_size = 16
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnetPD = resnet50(weights=ResNet50_Weights.DEFAULT)
resnetPD.fc = nn.Identity()
resnetPD.to(device)
SuperFinal = nn.Sequential(nn.Linear(2048, 20))
SubFinal = nn.Sequential(nn.Linear(2048, 6))
SuperFinal.load_state_dict(torch.load('SuperFinal.pth'))
SubFinal.load_state_dict(torch.load('SubFinal.pth'))
brain_net = BrainNet(SuperFinal, SubFinal, resnetPD)
# Save the results in a latex formatted table in a file called table.tex
table_data = []
sigm = nn.Softmax()
for data, target in iter(data_loader):
    data = data.to(device)
    target = target.to(device)
    output1, output2 = brain_net(data)
    target1 = ((target - 1) // 5).clamp(0, 19)
    #confidence of output1
    prbs1 = F.softmax(output1, dim=1)
    predicted_class1 = torch.argmax(prbs1, dim=1)
    conf1 = prbs1[torch.arange(len(predicted_class1)), predicted_class1]
    #confidence of output2
    prbs2 = F.softmax(output2, dim=1)
    predicted_class2 = torch.argmax(prbs2, dim=1)
    conf2 = prbs2[torch.arange(len(predicted_class2)), predicted_class2]
    for i in range(len(target)):
        row = [
            output1[i].argmax(dim=0).item(),
            conf1[i].item(),
            target1[i].item(),
            output2[i].argmax(dim=0).item(),
            conf2[i].item(),
            target[i].item()
        ]
        table_data.append(row)

table_latex = "\\begin{table}[ht]\n"
table_latex += "\\centering\n"
table_latex += "\\begin{tabular}{|c|c|c|c|c|c|}\n"
table_latex += "\\hline\n"
table_latex += "Output 1 & Confidence 1 & Target 1 & Output 2 & Confidence 2 & Target \\\\\n"
table_latex += "\\hline\n"

for row in table_data:
    formatted_row = [f"{val:.2f}" if isinstance(val, (float)) else str(val) for val in row]
    table_latex += " & ".join(formatted_row) + " \\\\\n"

table_latex += "\\hline\n"
table_latex += "\\end{tabular}\n"
table_latex += "\\caption{Your table caption here.}\n"
table_latex += "\\label{tab:your_table_label}\n"
table_latex += "\\end{table}\n"

with open("table.tex", "w") as f:
    f.write(table_latex)