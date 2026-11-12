#
# GELU (tanh form) at the SigLIP vision-encoder MLP shape used by SmolVLA:
# one 512 x 512 camera image = 1024 patch tokens, intermediate size 3072,
# hidden_act = gelu_pytorch_tanh (transformers SmolVLMVisionConfig defaults).
#
# Run: make acti-run acti=config/kernels/acti_gelu_siglip.py
#

class Activation:

    def __init__(self):

        self.dtype                   = 'fp16'
        self.algo                    = 'gelu'
        self.m_size                  = 1024
        self.n_size                  = 3072

        self.gate_enable             = 0
        self.bias_enable             = 0
