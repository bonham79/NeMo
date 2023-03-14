# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F

from nemo.core import typecheck
from nemo.collections.asr.losses.ssl_losses.contrastive import ContrastiveLoss



class ReconstructiveLoss(ContrastiveLoss):
    def __init__(self, *args, **dargs):
        super().__init__(*args, **dargs)
        self._loss = torch.nn.L1Loss(reduction='sum')
    @typecheck()
    def forward(self, spectrograms, spec_masks, decoder_outputs, decoder_lengths=None):
        spec_in = spectrograms.transpose(-2, -1)
        masks = spec_masks.transpose(-2, -1)
        targets = spec_in
        # BxTxC

        targets = targets.reshape(targets.shape[0], targets.shape[1] // self.combine_time_steps, -1)
        masks = masks.reshape(targets.shape[0], targets.shape[1], -1)

        bs = decoder_outputs.shape[0]
        masks = masks.mean(-1) > self.mask_threshold
        out_masked_only = decoder_outputs[masks]
        targets_masked_only = targets[masks]
        out_masked_only = out_masked_only.reshape(bs, -1, out_masked_only.shape[-1])
        targets_masked_only = targets_masked_only.reshape(bs, -1, targets_masked_only.shape[-1])

        # BxT'xC
        # number of masked time steps to predict (T')

        loss = self._loss(out_masked_only, targets_masked_only)

        if not isinstance(loss, torch.Tensor):
            loss = torch.Tensor([0]).to(device=decoder_outputs.device)

        return loss
