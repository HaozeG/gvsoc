#
# Copyright (C) 2025 ETH Zurich and University of Bologna
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
#

# LayerNorm over the last dimension: y = (x - mean) / sqrt(var + eps) * gamma + beta.
# Default: the SigLIP vision encoder in SmolVLA, one 512 x 512 camera image
# = 1024 patch tokens of width 768, layer_norm_eps = 1e-6
# (transformers SmolVLMVisionConfig defaults).

class LayerNorm:

    def __init__(self):

        self.dtype                   = 'fp16'   # fp16 only
        self.m_size                  = 1024     # tokens
        self.n_size                  = 768      # hidden size
        self.eps                     = 1e-6
