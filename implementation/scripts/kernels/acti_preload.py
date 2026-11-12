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

# Author: Chi Zhang <chizhang@ethz.ch>

import re
import os
import io
import sys
import math
import shutil
import argparse
import numpy as np
import importlib.util
from tqdm import tqdm
import preload as pld

def import_module_from_path(module_path):
    """
    Dynamically import a module from an absolute path and mimic `from module import *`.
    """
    module_name = os.path.splitext(os.path.basename(module_path))[0]  # Extract the file name without extension
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Cannot find a module at path: {module_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Mimic `from module import *`
    globals().update(vars(module))
    return module

def acti_inputs(m, n, seed=0):
    """fp16 test input: mostly N(0, 3^2), plus a sweep over [-60, 60] in every
    row so that each Spatz slice sees exp overflow, underflow and x^2 overflow."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 3.0, size=(m, n))
    sweep = np.linspace(-60.0, 60.0, 64)
    for r in range(m):
        cols = rng.choice(n, size=min(64, n), replace=False)
        x[r, cols] = sweep[:len(cols)]
    return x.astype(np.float16)


def acti_fp16_replay(algo, x):
    """Replay the kernel's instruction sequence in fp16 (one rounding per
    instruction; exp evaluated in float64 then rounded, as the GVSoC model does).
    This is the golden the device output is compared against."""
    h = np.float16
    with np.errstate(all="ignore"):
        if algo == "relu":
            return np.maximum(x, h(0))
        if algo in ("sigmoid", "silu"):
            one = h(1.0)
            mx = (x * h(-1.0)).astype(h)
            e = np.exp(mx.astype(np.float64)).astype(h)
            den = (e + one).astype(h)
            num = np.full_like(x, one) if algo == "sigmoid" else x
            return (num / den).astype(h)
        if algo == "gelu":
            c, one, m2k = h(0.044715), h(1.0), h(-1.5957691)
            t = (x * x).astype(h)
            t = (t * c).astype(h)
            t = (t + one).astype(h)
            t = (t * x).astype(h)
            t = (t * m2k).astype(h)
            e = np.exp(t.astype(np.float64)).astype(h)
            den = (e + one).astype(h)
            return (x / den).astype(h)
    raise ValueError(f"no replay for algo {algo}")


def acti_reference(algo, x):
    """The true function in float64 (gelu: tanh form, gelu_pytorch_tanh)."""
    x = x.astype(np.float64)
    with np.errstate(all="ignore"):
        if algo == "relu":
            return np.maximum(x, 0.0)
        if algo == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if algo == "silu":
            return x / (1.0 + np.exp(-x))
        if algo == "gelu":
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
    raise ValueError(f"no reference for algo {algo}")


def fp16_ulp_distance(a, b):
    def ordered(v):
        u = v.astype(np.float16).view(np.uint16).astype(np.int32)
        mag = u & 0x7FFF
        return np.where(u & 0x8000, -mag, mag)
    return np.abs(ordered(a) - ordered(b))


def report_algorithm_error(algo, x, golden):
    """How far the fp16 instruction sequence is from the true function.
    ULP distance is misleading where the output is near 0 (e.g. GELU of
    negative x), so report absolute error and relative error where |ref| >= 0.1."""
    ref = acti_reference(algo, x)
    ok = ~np.isnan(golden)
    g = golden[ok].astype(np.float64)
    r = ref[ok]
    err = np.abs(g - r)
    big = np.abs(r) >= 0.1
    rel = err[big] / np.abs(r[big])
    ulp = fp16_ulp_distance(golden[ok], r.astype(np.float16))
    print(f"[acti {algo}] fp16 sequence vs float64 reference over |x| <= {np.abs(x).max():.0f}: "
          f"max abs err {err.max():.2e}; max rel err {rel.max():.2e} where |ref| >= 0.1; "
          f"median {np.median(ulp):.0f} ULP; NaN outputs {int((~ok).sum())} of {golden.size}")


def gen_acti_preload_numpy_arrays(acti):
    if acti.dtype != "fp16":
        # fp8: no replay; keep the placeholder data (no numerical check)
        dtype = np.uint8
        I_np = np.ones((acti.m_size, acti.n_size), dtype=dtype)
        O_empty = np.zeros(I_np.shape, dtype=dtype)
        return I_np, I_np, I_np, O_empty, I_np

    x = acti_inputs(acti.m_size, acti.n_size)
    o = acti_fp16_replay(acti.algo, x)
    rng = np.random.default_rng(1)
    G = rng.normal(0.0, 1.0, size=x.shape).astype(np.float16)
    B = rng.normal(0.0, 1.0, size=x.shape).astype(np.float16)
    with np.errstate(all="ignore"):
        if acti.gate_enable:
            o = (G * o).astype(np.float16)
        if acti.bias_enable:
            o = (B + o).astype(np.float16)
    if not acti.gate_enable and not acti.bias_enable:
        report_algorithm_error(acti.algo, x, o)

    u16 = lambda a: a.view(np.uint16)
    return u16(x), u16(G), u16(B), np.zeros(x.shape, dtype=np.uint16), u16(o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate C header files from a Activation preload file and Dynamically import Python modules from absolute or relative paths.")
    parser.add_argument("elf_path", type=str, help="Path of preload elf file")
    parser.add_argument("header_path", type=str, help="Path to Activation preload C header file")
    parser.add_argument(
        "module_paths",
        metavar="module_path",
        type=str,
        nargs="+",
        help="Paths to Python modules to import (absolute or relative)."
    )

    args = parser.parse_args()

    # Process each provided module path
    for module_path in args.module_paths:
        # Convert to absolute path
        absolute_path = os.path.abspath(module_path)

        if not os.path.isfile(absolute_path):
            print(f"Error: {absolute_path} is not a valid file.")
            continue

        try:
            # Import the module dynamically
            module = import_module_from_path(absolute_path)
            print(f"Successfully imported: {module.__name__} from {absolute_path}")
        except Exception as e:
            print(f"Failed to import {absolute_path}: {e}")

    # Instanciate arch and acti configurations
    acti = Activation()
    arch = FlexClusterArch()

    # Generate the preload
    I_np, G_np, B_np, O_empty, O_golden = gen_acti_preload_numpy_arrays(acti)
    I_addr = arch.hbm_start_base
    O_eaddr = I_addr + I_np.nbytes
    O_gaddr = O_eaddr + O_empty.nbytes
    # The preload ELF is 32-bit: an address at or above 4 GiB is truncated and
    # lands on top of other arrays. The previous gate address,
    # hbm_start_base + hbm_node_addr_space * (2 * num_cluster_y + num_cluster_x)
    # = 0x9c0000000 on the default arch, truncated to 0xc0000000 = I_addr.
    # Place gate and bias after the golden instead.
    G_addr = O_gaddr + O_golden.nbytes
    B_addr = G_addr + G_np.nbytes
    assert B_addr + B_np.nbytes <= 1 << 32, "preload arrays must lie below 4 GiB (32-bit ELF)"
    print(f"I_addr = {I_addr: #x}")
    print(f"O_eaddr = {O_eaddr: #x}")
    print(f"O_gaddr = {O_gaddr: #x}")
    print(f"G_addr = {G_addr: #x}")
    print(f"B_addr = {B_addr: #x}")

    # Real data so that main.c can compare O_EADDR against O_GADDR.
    # Gate and bias arrays are written only when enabled.
    arrays, addrs = [I_np, O_empty, O_golden], [I_addr, O_eaddr, O_gaddr]
    if acti.gate_enable:
        arrays.append(G_np); addrs.append(G_addr)
    if acti.bias_enable:
        arrays.append(B_np); addrs.append(B_addr)
    pld.make_preload_elf(args.elf_path, arrays, addrs)

    #generate preload header file
    with open(args.header_path, 'w') as file:
        file.write('#ifndef _ACTI_PRELOAD_H_\n')
        file.write('#define _ACTI_PRELOAD_H_\n\n')
        file.write(f'#define {"I_addr".upper()} ((uint64_t){I_addr: #x})\n')
        file.write(f'#define {"O_eaddr".upper()} ((uint64_t){O_eaddr: #x})\n')
        file.write(f'#define {"O_gaddr".upper()} ((uint64_t){O_gaddr: #x})\n')
        file.write(f'#define {"G_addr".upper()} ((uint64_t){G_addr: #x})\n')
        file.write(f'#define {"B_addr".upper()} ((uint64_t){B_addr: #x})\n')
        file.write('\n#endif // _ACTI_PRELOAD_H_\n')

    print(f'Header file "{args.header_path}" generated successfully.')
