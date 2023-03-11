import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision

class resnet18(nn.Module):
    def __init__(self, num_classes):
        super(resnet18, self).__init__()
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.net.fc = nn.Linear(in_features = 512, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.net(x)
        return x

class resnet50(nn.Module):
    def __init__(self, num_classes):
        super(resnet50, self).__init__()
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.net.fc = nn.Linear(in_features = 2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.net(x)
        return x

class swinT(nn.Module):
    def __init__(self, num_classes):
        super(swinT, self).__init__()
        self.net = torchvision.models.swin_t(weights='DEFAULT')
        self.net.head = nn.Linear(in_features = 768, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.net(x)
        return x

class swinS(nn.Module):
    def __init__(self, num_classes):
        super(swinS, self).__init__()
        self.net = torchvision.models.swin_s(weights='DEFAULT')
        self.net.head = nn.Linear(in_features = 768, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.net(x)
        return x

class swinB(nn.Module):
    def __init__(self, num_classes):
        super(swinB, self).__init__()
        self.net = torchvision.models.swin_b(weights='DEFAULT')
        self.net.head = nn.Linear(in_features = 1024, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.net(x)
        return x
    
class mobilenet_v2(nn.Module):
    def __init__(self, num_classes):
        super(mobilenet_v2, self).__init__()
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.net.classifier[-1] = nn.Linear(in_features = 1280, out_features=num_classes, bias=True)
    
    def forward(self, x):
        x = self.net(x)
        return x
