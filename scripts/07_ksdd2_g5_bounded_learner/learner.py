"""The only learner configuration authorized by KSDD2 G5."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        groups: int,
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, groups)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        inputs = self.reduce(inputs)
        return self.conv(torch.cat((skip, inputs), dim=1))


class CompactUNet(nn.Module):
    def __init__(self, channels: list[int], groups: int = 8) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError("Frozen compact U-Net requires exactly five channel levels")
        if any(channel % groups for channel in channels):
            raise ValueError("Every frozen channel width must be divisible by GroupNorm groups")
        c0, c1, c2, c3, c4 = channels
        self.enc0 = DoubleConv(3, c0, groups)
        self.enc1 = DoubleConv(c0, c1, groups)
        self.enc2 = DoubleConv(c1, c2, groups)
        self.enc3 = DoubleConv(c2, c3, groups)
        self.bottleneck = DoubleConv(c3, c4, groups)
        self.pool = nn.MaxPool2d(2)
        self.up3 = UpBlock(c4, c3, c3, groups)
        self.up2 = UpBlock(c3, c2, c2, groups)
        self.up1 = UpBlock(c2, c1, c1, groups)
        self.up0 = UpBlock(c1, c0, c0, groups)
        self.head = nn.Conv2d(c0, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(inputs)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        decoded = self.up3(bottleneck, e3)
        decoded = self.up2(decoded, e2)
        decoded = self.up1(decoded, e1)
        decoded = self.up0(decoded, e0)
        return self.head(decoded)


def build_model(config: dict[str, Any]) -> CompactUNet:
    learner = config["learner"]
    if learner["name"] != "compact_unet_binary_segmentation":
        raise RuntimeError("Only the frozen compact U-Net is permitted")
    return CompactUNet(
        channels=[int(value) for value in learner["encoder_channels"]],
        groups=int(learner["group_norm_groups"]),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_config = config["training"]["loss"]
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    epsilon = float(loss_config["soft_dice_epsilon"])
    soft_dice_loss = (1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon))).mean()
    total = (
        float(loss_config["bce_with_logits_weight"]) * bce
        + float(loss_config["soft_dice_weight"]) * soft_dice_loss
    )
    return total, {"bce": float(bce.detach()), "soft_dice": float(soft_dice_loss.detach())}

