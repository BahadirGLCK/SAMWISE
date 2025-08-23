# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Backbones from the TIMM library (RepViT, etc.)."""

from typing import List, Tuple

import torch
from torch import nn

from timm.models import create_model


class TimmBackbone(nn.Module):
    def __init__(
        self,
        name: str,
        features: Tuple[str, ...] = ("layer0", "layer1", "layer2", "layer3"),
    ):
        super().__init__()

        out_indices = tuple(int(f[len("layer") :]) for f in features)

        backbone = create_model(
            name,
            pretrained=True,
            in_chans=3,
            features_only=True,
            out_indices=out_indices,
        )

        num_channels = backbone.feature_info.channels()
        # FPN expects channel_list in high->low order
        self.channel_list = num_channels[::-1]
        self.body = backbone

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        xs = self.body(x)
        # Return list of 4 feature maps from low->high resolution order as expected by FPN
        return list(xs)


