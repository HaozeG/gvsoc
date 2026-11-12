#
# Eltwise: elementwise row operators on Spatz.
#   op = 'add'   y = x + z            (residual add, position-embedding add)
#        'bias'  y = x + b            (b: one row of n_size, broadcast)
#        'scale' y = alpha x
#        'axpy'  y = x + alpha z      (flow-matching Euler update: alpha = dt)
#        'rope'  rotate-half RoPE per head of head_dim columns, table (cos | sin) per row
#        'pixel_shuffle'  SmolVLM connector: m_size output rows of n_size = pix_scale^2 x D from
#                 images of grid x grid patch rows of D (DMA only, no Spatz)
# Run: make elt-run elt=config/kernels/elt.py
#

class Eltwise:

    def __init__(self):

        self.dtype                   = 'fp16'
        self.op                      = 'rope'
        self.m_size                  = 64
        self.n_size                  = 960
        self.alpha                   = -0.1
        self.head_dim                = 64
        self.grid                    = 32
        self.pix_scale               = 4
