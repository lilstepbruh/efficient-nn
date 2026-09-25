from collections import OrderedDict

from torch import nn


class SmallCNN(nn.Sequential):
    """Map FP32 images [B, 3, S, S] to logits [B, 100].

    Homework inputs have S divisible by 16. Convolutions have no bias;
    both linear layers explicitly include bias. ReLU is always in-place.
    For measurements, use model.float().to(device).eval() together with
    torch.inference_mode(); the module itself does not alter global settings.
    """

    def __init__(self):
        super().__init__(OrderedDict([
            ("conv1", nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool", nn.MaxPool2d(3, stride=2, padding=1)),
            ("conv2", nn.Conv2d(32, 64, 5, padding=2, bias=False)),
            ("relu2", nn.ReLU(inplace=True)),
            ("conv3", nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(128, 256, 1, padding=0, bias=False)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False)),
            ("relu5", nn.ReLU(inplace=True)),
            ("conv6", nn.Conv2d(256, 512, 1, padding=0, bias=False)),
            ("relu6", nn.ReLU(inplace=True)),
            ("avgpool", nn.AdaptiveAvgPool2d(1)),
            ("flatten", nn.Flatten(start_dim=1)),
            ("fc1", nn.Linear(512, 256, bias=True)),
            ("relu7", nn.ReLU(inplace=True)),
            ("fc2", nn.Linear(256, 100, bias=True)),
        ]))
