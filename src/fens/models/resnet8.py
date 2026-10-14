"""
ResNet-8 architecture for 32x32 vision tasks (CIFAR-10, CIFAR-100, SVHN).

Paper Reference:
  - Section 3.1 (Configurations): "In line with related work [10, 24, 35], we
    use ResNet-8 [13] as the client local model for our vision tasks."
  - Section 3.7 & Table 4 (Memory Footprint Analysis):
    * Standard `resnet8` uses channel widths (64, 128, 256), yielding ~4.9M
      FP32 parameters (~18.85 MiB per model, or 377.03 MiB for a 20-client
      unquantized ensemble, matching Table 4).
    * Downsized `resnet8_ds` reduces layer widths to (14, 28, 56) / (16, 32, 64),
      yielding ~0.88 MiB per model so a 20-client ensemble + aggregator is
      ~17.95 MiB (matching a single standard ResNet-8 in FedAdam, Table 4).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard 2-conv residual block for CIFAR/SVHN ResNet-8."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet8(nn.Module):
    """
    ResNet-8 architecture for 32x32 images:
      - 1 initial Conv2dstem
      - 3 residual stages with 1 BasicBlock (2 convs) each -> 1 + 2*3 + 1 FC = 8 layers.

    Args:
        num_classes: Number of output classes C (default 10).
        channels: Tuple of 3 channel widths for stages 1, 2, and 3.
            - Standard ResNet-8 (`resnet8`): `(64, 128, 256)` (~18.85 MiB in FP32, Table 4)
            - Downsized ResNet-8 (`resnet8_ds`): `(14, 28, 56)` (~0.88 MiB in FP32, Table 4)
    """

    def __init__(
        self,
        num_classes: int = 10,
        channels: Tuple[int, int, int] = (64, 128, 256),
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.in_planes = c1

        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)

        # 3 stages, 1 BasicBlock per stage -> 6 conv layers + 1 stem conv + 1 linear = 8 layers
        self.layer1 = self._make_layer(c1, num_blocks=1, stride=1)
        self.layer2 = self._make_layer(c2, num_blocks=1, stride=2)
        self.layer3 = self._make_layer(c3, num_blocks=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3 * BasicBlock.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw unnormalized class logits z_i in R^C."""
        feats = self.forward_features(x)
        logits = self.fc(feats)
        return logits


def build_client_model(arch: str = "resnet8", num_classes: int = 10) -> nn.Module:
    """
    Factory function to construct a client model:
      - `resnet8`: Standard ResNet-8 with channels (64, 128, 256)
      - `resnet8_ds`: Downsized ResNet-8 from Section 3.7 / Table 4 with channels (14, 28, 56)
      - `resnet8_light`: Lightweight ResNet-8 with channels (16, 32, 64) for ultra-fast CPU/GPU testing
    """
    arch = arch.lower()
    if arch == "resnet8":
        return ResNet8(num_classes=num_classes, channels=(64, 128, 256))
    elif arch == "resnet8_ds":
        return ResNet8(num_classes=num_classes, channels=(14, 28, 56))
    elif arch == "resnet8_light":
        return ResNet8(num_classes=num_classes, channels=(16, 32, 64))
    else:
        raise ValueError(
            f"Unknown architecture '{arch}'. Choose 'resnet8', 'resnet8_ds', or 'resnet8_light'."
        )


def get_model_size_bytes(model: nn.Module, bits_per_param: int = 32) -> int:
    """Computes the parameter footprint of `model` in bytes at `bits_per_param` precision."""
    num_params = sum(p.numel() for p in model.parameters())
    return int(num_params * (bits_per_param / 8))


def get_model_size_mib(model: nn.Module, bits_per_param: int = 32) -> float:
    """Computes the parameter footprint of `model` in MiB (1 MiB = 2^20 bytes)."""
    return get_model_size_bytes(model, bits_per_param=bits_per_param) / (1024.0 ** 2)
