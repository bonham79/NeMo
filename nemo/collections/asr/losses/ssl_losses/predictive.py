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
from nemo.collections.asr.losses.ssl_losses.contrastive import ContrastiveLoss
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, LossType, NeuralType, SpectrogramType

__all__ = ["ContrastiveLoss"]


class PredictiveLoss(ContrastiveLoss):
 class ContrastiveLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for Contrastive.
        """
        return {
            "spectrograms": NeuralType(("B", "D", "T"), SpectrogramType()),
            "spec_masks": NeuralType(("B", "D", "T"), SpectrogramType()),
            "decoder_outputs": NeuralType(("B", "T", "D"), AcousticEncodedRepresentation()),
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
        self,
        in_dim: int,
        proj_dim: int = 128,
        combine_time_steps: int = 1,
        num_negatives: int = 100,
        quantized_targets: bool = False,
        codebook_size: int = 320,
        prob_ppl_weight: float = 0.1,
        logit_temp: float = 0.1,
        reduce: str = "sum",
        sample_from_same_utterance_only: bool = True,
        sample_from_non_masked: bool = False,
        sample_from_codebook: bool = False,
        group_loss: bool = False,
        num_groups: int = 2,
        quantizer_temp_start: float = 2,
        quantizer_temp_min: float = 0.5,
        quantizer_temp_decay: float = 0.999995,
        mask_threshold: float = 0.8,
        store_ids: bool = True,
        reduce_ids: bool = False,
        multiplier: float = 16.0,
    ):
        """
        Loss function representing the contrastive task of identifying the true latent speech representation of
        the masked spectrogram steps from a set of sampled distractors.

        Args:
            in_dim: Number of spectrogram channels.
            proj_dim: Number of channels in the model outputs.
            combine_time_steps: How many time steps should be combined into a single representation.
            num_negatives: Number of sampled negatives for each target.
            quantized_targets: Bool that determines if the targets should be quantized.
            codebook_size: Number of vectors in the codebook per group.
            prob_ppl_weight: Float multiplier on the perplexity loss for target quantization.
            logit_temp: Float temperature for normalizing logits.
            reduce: String representing the type of reduction used for cross entropy.
            sample_from_same_utterance_only: Bool that determines if negatives should be sampled only from same utterance.
            sample_from_non_masked: Bool that determines if negatives should be sampled from non-masked steps of the spectrogram.
            sample_from_codebook: Bool that determines if negatives should be sampled from entire codebook.
            group_loss: Bool that determines if loss should be computed separately for each group in the quantizer codebook.
            num_groups: Number of groups in the quantizer codebook.
            quantizer_temp_start: Starting temperature in quantizer.
            quantizer_temp_min: Minimum temperature in quantizer.
            quantizer_temp_decay: Decay rate of quantizer temperature per global step.
            mask_threshold: Float threshold for determining if a time step of the spectrogram is masked based on percent of masked channels.
            store_ids: Bool that determines if the quantizer ids will be stored to be potentially used by other losses.
            reduce_ids: Bool that determines if we convert any sequence of consecutive equivalent ids to a single occurence of that id.
            multiplier: Float multipler on final loss
        """

        super().__init__()
        quantizer_temp = (quantizer_temp_start, quantizer_temp_min, quantizer_temp_decay)
        self.quantized_targets = quantized_targets
        self.num_negatives = num_negatives
        self.prob_ppl_weight = prob_ppl_weight
        if self.quantized_targets:
            quantizer_cfg = {
                "_target_": "nemo.collections.asr.parts.submodules.ssl_quantizers.GumbelClassQuantizer",
                "dim": in_dim * combine_time_steps,
                "vq_dim": proj_dim,
                "num_vars": codebook_size,
                "groups": num_groups,
                "temp": quantizer_temp,
                "combine_groups": True,
                "time_first": True,
            }
            self.quantizer = ContrastiveLoss.from_config_dict(quantizer_cfg)
        self.prob_ppl_weight = prob_ppl_weight
        self.logit_temp = logit_temp
        self.reduce = reduce
        self.combine_time_steps = combine_time_steps
        self.sample_from_same_utterance_only = sample_from_same_utterance_only
        self.sample_from_non_masked = sample_from_non_masked
        self.sample_from_codebook = sample_from_codebook
        self.group_loss = group_loss
        self.mask_threshold = mask_threshold
        self.multiplier = multiplier

        self.store_ids = store_ids
        self.reduce_ids = reduce_ids

        if not self.quantized_targets:
            self.target_proj = nn.Linear(in_dim * combine_time_steps, proj_dim)

    @typecheck()
    def forward(self, spectrograms, spec_masks, decoder_outputs, decoder_lengths=None):
        spec_in = spectrograms.transpose(-2, -1)
        masks = spec_masks.transpose(-2, -1)
        targets = spec_in
        # BxTxC

        targets = targets.reshape(targets.shape[0], targets.shape[1] // self.combine_time_steps, -1)
        masks = masks.reshape(targets.shape[0], targets.shape[1], -1)

        var_probs, prob_ppl_loss, cur_codebook_temp = self.quantizer(targets)
        # -> BxTxGxC

        loss = torch.sum(torch.log(var_probs * targets))

        sample_size = torch.sum(masks)

        if self.prob_ppl_weight != 0 and self.quantized_targets:
            prob_ppl_loss = self.prob_ppl_weight * prob_ppl_loss * sample_size
            loss += prob_ppl_loss

        if not isinstance(loss, torch.Tensor):
            loss = torch.Tensor([0]).to(device=decoder_outputs.device)

        return loss
