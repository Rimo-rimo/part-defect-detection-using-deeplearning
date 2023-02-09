import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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

# if CFG["model"] == "resnet18":
#     model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#     model.fc = nn.Linear(in_features = 512, out_features=4, bias=True)
# elif CFG["model"] == "resnet50":
#     model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
#     model.fc = nn.Linear(in_features = 2048, out_features=4, bias=True)