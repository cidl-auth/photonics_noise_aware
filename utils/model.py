import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as T

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GaussianNoise:
    def __init__(self, std=None):
        self.std = std

    def __call__(self, x, layer=None):
        if self.std is not None and self.std != 0.0:
            range = T.std(x, dim=0, keepdim=True)
            noise = T.randn(x.size(), device=DEVICE) * self.std * range
            x = x + noise.to(DEVICE)
        return x


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()

        self.noise_module = GaussianNoise(None)

        self.conv1 = nn.Conv2d(1, 64, 3, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, bias=False)

        self.fc_red = nn.Linear(9216, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

        # Define the possible acitvations that can be used
        self.software_activation = lambda x: torch.abs(torch.sin(x.clamp(0.0, 1.0) ** 2 * math.pi / 2.0))
        self.reduction_activation = lambda x: torch.abs(torch.sin(x.clamp(0.0, 1.0) ** 2 * math.pi / 2.0))
        self.photonic_activation = lambda x: torch.abs(torch.sin(x.clamp(0.0, 1.0) ** 2 * math.pi / 2.0))
        self.final_activation = lambda x: x ** 2

    def set_noise_level(self, noise_level):
        self.noise_module.std = noise_level

    def get_weight_loss(self, alpha=10):
        w = self.fc1.weight
        weights_to_clip = torch.sum(w[torch.abs(w) > 1] ** 2)
        w = self.fc2.weight
        weights_to_clip += torch.sum(w[torch.abs(w) > 1] ** 2)
        w = self.conv1.weight
        weights_to_clip += torch.sum(w[torch.abs(w) > 1] ** 2)
        w = self.conv2.weight
        weights_to_clip += torch.sum(w[torch.abs(w) > 1] ** 2)
        w = self.fc_red.weight
        weights_to_clip += torch.sum(w[torch.abs(w) > 1] ** 2)

        return alpha * weights_to_clip

    def apply_constraints(self):
        self.conv1.weight.data = torch.clip(self.conv1.weight.data, -1, 1)
        self.conv2.weight.data = torch.clip(self.conv2.weight.data, -1, 1)
        self.fc_red.weight.data = torch.clip(self.fc_red.weight.data, -1, 1)
        self.fc1.weight.data = torch.clip(self.fc1.weight.data, -1, 1)
        self.fc2.weight.data = torch.clip(self.fc2.weight.data, -1, 1)

    def forward(self, x):
        x = self.noise_module(x)
        x = self.conv1(x)
        x = self.software_activation(x)

        x = self.noise_module(x)
        x = self.conv2(x)
        x = self.software_activation(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc_red(x)
        x = self.reduction_activation(x)

        x = self.noise_module(x)
        x = self.fc1(x)
        x = self.photonic_activation(x)

        x = self.noise_module(x)
        x = self.fc2(x)
        x = self.final_activation(x)

        return x
