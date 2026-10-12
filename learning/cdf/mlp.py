"""Neural network used by the pretrained Franka CDF model.

Adapted from https://github.com/yimingli1998/cdf (MIT license).
"""
from __future__ import annotations

import torch
from torch import nn


def _mlp(channels: list[int]) -> nn.Sequential:
    layers = [
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
        for i in range(1, len(channels) - 1)
    ]
    layers.append(nn.Sequential(nn.Linear(channels[-2], channels[-1])))
    return nn.Sequential(*layers)


class MLPRegression(nn.Module):
    """Architecture matching the supplied Panda CDF checkpoint."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_mlp([30, 1024, 512, 256, 128, 128, 1])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = torch.cat((x, torch.sin(x), torch.cos(x)), dim=-1)
        return self.layers[0](encoded)
