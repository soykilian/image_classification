from torch.utils.data import TensorDataset
import torch

def FreezeModel(model):
    for par in model.parameters():
        par.requires_grad=False         # Will no longer learn this parameter

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

