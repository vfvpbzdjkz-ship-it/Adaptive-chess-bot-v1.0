"""Residual CNN with policy + value heads."""
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ouroboros.engine.encoding import POLICY_SIZE

log = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class OuroborosNet(nn.Module):
    """Policy + value residual network."""

    def __init__(self, blocks: int = 5, channels: int = 64, in_planes: int = 19):
        super().__init__()
        self.blocks = blocks
        self.channels = channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res_tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 64, POLICY_SIZE)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.res_tower(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        logits = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return logits, v


def build_net(cfg: dict, device: str) -> OuroborosNet:
    net = OuroborosNet(
        blocks=cfg.get("net_blocks", 5),
        channels=cfg.get("net_channels", 64),
    )
    return net.to(device)


def save_checkpoint(net: OuroborosNet, path: Path, metadata: Optional[dict] = None) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save({"state_dict": net.state_dict(), "metadata": metadata or {}}, tmp)
    import os
    os.replace(tmp, path)
    log.debug("Checkpoint saved → %s", path)


def load_checkpoint(
    net: OuroborosNet, path: Path, device: str, strict: bool = True
) -> dict:
    data = torch.load(str(path), map_location=device, weights_only=False)
    net.load_state_dict(data["state_dict"], strict=strict)
    return data.get("metadata", {})


def best_path() -> Path:
    return MODELS_DIR / "best.pt"


def latest_path() -> Path:
    return MODELS_DIR / "latest.pt"


def ckpt_path(step: int) -> Path:
    return MODELS_DIR / f"ckpt_{step}.pt"
