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
from torch import nn

from nemo.core import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType, SpectrogramType, AcousticEncodedRepresentation

from nemo.collections.asr.parts.submodules.ssl_quantizers import RandomQuantizer



class FixedCodebookPredictiveLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for Contrastive.
        """
        return {
            "spectrograms": NeuralType(("B", "D", "T"), SpectrogramType()),
            "spec_masks": NeuralType(("B", "D", "T"), SpectrogramType()),
            "decoder_outputs": NeuralType(("B", "T", "W", "D"), LogprobsType()),
            "decoder_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        """Output types definitions for Contrastive.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    @property
    def needs_labels(self):
        return False

    def __init__(
        self, combine_time_steps: int = 1, mask_threshold: float = 0.8, num_groups=2,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.nll_loss = nn.ModuleList(nn.NLLLoss() for _ in range(num_groups))
        self.combine_time_steps = combine_time_steps
        self.mask_threshold = mask_threshold

        self.quantized_targets = True
        self.store_ids= True
        self.sample_from_same_utterance_only=True

    @typecheck()
    def forward(self, spectrograms, spec_masks, decoder_outputs, decoder_lengths=None):
        bsz, _, tsz  = spectrograms.shape

        spec_in = spectrograms.transpose(-2, -1)
        masks = spec_masks.transpose(-2, -1)
        targets = spec_in
        # BxTxC

        targets = targets.reshape(bsz, tsz // self.combine_time_steps, -1)
        masks = masks.reshape(bsz, tsz // self.combine_time_steps, -1)

        _, target_ids = self.quantizer(targets, return_ids=True)
        # -> BxTxC, BxTxG

        targets = target_ids
        decoder_outputs = decoder_outputs.transpose(-1,-2)
        # -> BxTxCxG

        masks = masks.mean(-1) > self.mask_threshold
        out_masked_only = decoder_outputs[masks]
        # -> BT'xCxG
        targets_masked_only = targets[masks]
        # -> BT'xG

        total_loss = torch.zeros(1, device=targets_masked_only.device)
        for g in range(self.num_groups):
            loss = self.nll_loss[g](out_masked_only[:,:,g], targets_masked_only[:,g])
            total_loss += torch.mean(loss)

        return total_loss
