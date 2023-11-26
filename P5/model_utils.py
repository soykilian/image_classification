
from torch.utils.data import TensorDataset
import torch

def FreezeModel(model):
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
            y = ((y - 1) // 5 + 1).to(torch.int64)
            labels = y.cpu()
        else:
            features = torch.cat(
                [features, output.detach().cpu()])
            y = ((y - 1) // 5 + 1).to(torch.int64)
            labels = torch.cat([labels, y.cpu()])

    dataset = TensorDataset(features, labels)
    return dataset


def PreprocessedDatasetC(model, loader, device=None):
    if device is None:  
        device = next(model.parameters()).device
        features = None
        labels = None
 
    for i, (x, y) in enumerate(loader):
        model.eval()
        output = model(x.to(device))
        if i == 0:
            features = output.detach().cpu()
            y = ((y - 1) // 5).clamp(0, 19)
            labels = y.cpu()
        else:
            features = torch.cat(
                [features, output.detach().cpu()])
            y = ((y - 1) // 5).clamp(0, 19)
            labels = torch.cat([labels, y.cpu()])

    dataset = TensorDataset(features, labels)
    return dataset

class BrainNet(nn.Module):
    def __init__(self, SuperFinal, SubFinal, resnet):
        super(BrainNet, self).__init__()
        self.SuperFinal = SuperFinal
        self.SubFinal = SubFinal
        self.resnet = resnet

    def forward(self, x):
        x = self.resnet(x)
        x1 = self.SuperFinal(x)
        x2 = self.SubFinal(x)
        return x1, x2