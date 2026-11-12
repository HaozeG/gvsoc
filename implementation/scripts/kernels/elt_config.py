#
# Generate sw/Eltwise/include/elt.h from an Eltwise kernel configuration (config/kernels/elt.py).
# alpha (float) is written as its fp16 bit pattern ELT_ALPHA_H, since the generic header writer
# casts numbers to uint64_t.
#

import os
import argparse
import importlib.util
import numpy as np
import utils.kernel_configuration as kc


def import_module_from_path(module_path):
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Cannot find a module at path: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(vars(module))
    return module


parser = argparse.ArgumentParser(description="Generate the C header of an Eltwise configuration.")
parser.add_argument("input_file", help="Path to the Eltwise configuration python file")
parser.add_argument("output_file", help="Path to the Eltwise configuration C header file")
args = parser.parse_args()
import_module_from_path(os.path.abspath(args.input_file))

elt = Eltwise()
assert elt.dtype == "fp16", "Eltwise supports fp16 only"
assert elt.op in ("add", "bias", "scale", "axpy", "rope", "pixel_shuffle"), elt.op
assert elt.n_size % 4 == 0, "a row is split over 4 Spatz cores"
if elt.op == "pixel_shuffle":
    g = elt.grid // elt.pix_scale
    assert elt.grid % elt.pix_scale == 0 and elt.n_size % (elt.pix_scale ** 2) == 0 and elt.m_size % (g * g) == 0
if elt.op == "rope":
    assert elt.head_dim % 8 == 0 and elt.n_size % elt.head_dim == 0 and elt.head_dim // 2 <= 1024
alpha_h = int(np.array(elt.alpha, dtype=np.float16).view(np.uint16))
del elt.alpha
elt.alpha_h = alpha_h
kc.generate_config_C_header("ELT", elt, args.output_file, elt.dtype, 1)
