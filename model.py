import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from torchvision import models


class VGG(nn.Module):
    def __init__(self, args):
        super(VGG, self).__init__()

        self.cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }
        self.features = self._make_layers(self.cfg[args.vgg_name])
        self.classifier = nn.Linear(512, args.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        ipdb.set_trace()
        out = self.classifier(out)
        return F.softmax(out, dim=1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def make_network(args):
    if args.network == 'resnet18':
        network = models.resnet18(pretrained=True)
    elif args.network == 'resnet34':
        network = models.resnet34(pretrained=True)
    elif args.network == 'resnet50':
        network = models.resnet50(pretrained=True)
    elif args.network == 'resnet101':
        network = models.resnet101(pretrained=True)
    else:
        raise ValueError('=== Please check the proper network for training...')

    network.fc = torch.nn.Sequential(
        torch.nn.Linear(network.fc.in_features, args.num_classes),
        torch.nn.Softmax(dim=1))
    return network
