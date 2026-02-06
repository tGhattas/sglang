# Copyright 2025 SGLang Team
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

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(kw_only=True)
class Mamba1Metadata(ForwardMetadata):
    """Metadata for mamba1 mixer forward pass."""

    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    has_initial_states: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[list[int]] = None
    is_target_verify: bool = False
    draft_token_num: int = 1

    @staticmethod
    def prepare_decode(
        forward_metadata: ForwardMetadata,
        seq_lens: torch.Tensor,
        *,
        is_target_verify: bool,
        draft_token_num: int,
    ) -> "Mamba1Metadata":
        return Mamba1Metadata(
            query_start_loc=forward_metadata.query_start_loc,
            mamba_cache_indices=forward_metadata.mamba_cache_indices,
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
            num_decodes=len(seq_lens),
            num_prefills=0,
            num_prefill_tokens=0,
            is_target_verify=is_target_verify,
            draft_token_num=draft_token_num,
        )

    @classmethod
    def prepare_mixed(
        cls,
        forward_metadata: ForwardMetadata,
        forward_batch: ForwardBatch,
    ) -> "Mamba1Metadata":
        if forward_batch.extend_num_tokens is None:
            draft_token_num = (
                forward_batch.spec_info.draft_token_num
                if forward_batch.spec_info is not None
                else 1
            )
            return cls.prepare_decode(
                forward_metadata,
                forward_batch.seq_lens,
                is_target_verify=forward_batch.forward_mode.is_target_verify(),
                draft_token_num=draft_token_num,
            )

        num_prefills = len(forward_batch.extend_seq_lens)
        num_prefill_tokens = forward_batch.extend_num_tokens
        num_decodes = len(forward_batch.seq_lens) - num_prefills
        context_lens_tensor = forward_batch.extend_prefix_lens
        assert context_lens_tensor is not None
        has_initial_states = context_lens_tensor > 0
        return Mamba1Metadata(
            query_start_loc=forward_metadata.query_start_loc,
            mamba_cache_indices=forward_metadata.mamba_cache_indices,
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            has_initial_states=has_initial_states,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
            draft_token_num=(
                forward_batch.spec_info.draft_token_num
                if forward_batch.spec_info is not None
                else 1
            ),
        )
