#
# Preload for the Eltwise kernel: real fp16 inputs and a golden that replays the kernel's
# instruction sequence (one rounding per instruction; vfmacc rounds once).
# Arrays are placed from hbm_start_base upward and must stay below 4 GiB (32-bit preload ELF).
#

import os
import argparse
import importlib.util
import numpy as np
import preload as pld

H = np.float16
F = np.float64


def import_module_from_path(module_path):
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Cannot find a module at path: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(vars(module))
    return module


def r(a):
    """Round a float64 result to fp16 once (exact for sums and products of two fp16 values)."""
    return np.asarray(a, dtype=F).astype(H)


def rope_table(m, dh, theta=10000.0):
    """(cos | sin) per row; row i has position id i (lerobot apply_rope, max_wavelength 10000)."""
    inv = theta ** (-np.arange(0, dh, 2, dtype=F) / dh)
    ang = np.arange(m, dtype=F)[:, None] * inv[None, :]
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=1).astype(H)


def pixel_shuffle_gather(x, grid, scale):
    """SmolVLM connector pixel shuffle per image: output token (gr, gb) = the scale x scale patch
    block, row-major (== softhier-mlir frontend/smolvla.py pixel_shuffle)."""
    d = x.shape[1]
    g = grid // scale
    blk = x.reshape(-1, g, scale, g, scale, d).transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(blk.reshape(-1, scale * scale * d))


def elt_inputs(elt):
    rng = np.random.default_rng(0)
    m, n = elt.m_size, elt.n_size
    if elt.op == "pixel_shuffle":
        s = elt.pix_scale
        return rng.normal(0.0, 2.0, size=(m * s * s, n // (s * s))).astype(H), None
    x = rng.normal(0.0, 2.0, size=(m, n)).astype(H)
    if elt.op in ("add", "axpy"):
        z = rng.normal(0.0, 2.0, size=(m, n)).astype(H)
    elif elt.op == "bias":
        z = rng.normal(0.0, 1.0, size=(1, n)).astype(H)
    elif elt.op == "rope":
        z = rope_table(m, elt.head_dim)
    else:
        z = None
    return x, z


def elt_fp16_replay(elt, x, z):
    a = F(H(elt.alpha))
    with np.errstate(all="ignore"):
        if elt.op == "pixel_shuffle":
            return pixel_shuffle_gather(x, elt.grid, elt.pix_scale)
        if elt.op in ("add", "bias"):
            return r(x.astype(F) + z.astype(F))
        if elt.op == "scale":
            return r(a * x.astype(F))
        if elt.op == "axpy":
            return r(x.astype(F) + a * z.astype(F))
        if elt.op == "rope":
            dh, half = elt.head_dim, elt.head_dim // 2
            nh = elt.n_size // dh
            xh = x.reshape(elt.m_size, nh, dh)
            x1, x2 = xh[:, :, :half].astype(F), xh[:, :, half:].astype(F)
            c, s = z[:, None, :half].astype(F), z[:, None, half:].astype(F)
            y1 = r(r(x1 * c).astype(F) - r(x2 * s).astype(F))
            y2 = r(r(x2 * c).astype(F) + r(x1 * s).astype(F))
            return np.concatenate([y1, y2], axis=2).reshape(elt.m_size, elt.n_size)
    raise ValueError(elt.op)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate the Eltwise preload ELF and header.")
    parser.add_argument("elf_path", type=str, help="Path of preload elf file")
    parser.add_argument("header_path", type=str, help="Path to the Eltwise preload C header file")
    parser.add_argument("module_paths", metavar="module_path", type=str, nargs="+",
                        help="Arch and kernel configuration python files")
    args = parser.parse_args()
    for module_path in args.module_paths:
        import_module_from_path(os.path.abspath(module_path))

    elt = Eltwise()
    arch = FlexClusterArch()

    x, z = elt_inputs(elt)
    golden = elt_fp16_replay(elt, x, z)
    u16 = lambda a: a.view(np.uint16)
    I_addr = arch.hbm_start_base
    O_eaddr = I_addr + x.nbytes
    O_gaddr = O_eaddr + golden.nbytes
    Z_addr = O_gaddr + golden.nbytes
    end = Z_addr + (z.nbytes if z is not None else 0)
    assert end <= 1 << 32, "preload arrays must lie below 4 GiB (32-bit ELF)"
    arrays, addrs = [u16(x), np.zeros(golden.shape, dtype=np.uint16), u16(golden)], [I_addr, O_eaddr, O_gaddr]
    if z is not None:
        arrays.append(u16(z)); addrs.append(Z_addr)
    pld.make_preload_elf(args.elf_path, arrays, addrs)

    with open(args.header_path, 'w') as file:
        file.write('#ifndef _ELT_PRELOAD_H_\n#define _ELT_PRELOAD_H_\n\n')
        file.write(f'#define I_ADDR ((uint64_t){I_addr: #x})\n')
        file.write(f'#define O_EADDR ((uint64_t){O_eaddr: #x})\n')
        file.write(f'#define O_GADDR ((uint64_t){O_gaddr: #x})\n')
        file.write(f'#define Z_ADDR ((uint64_t){Z_addr: #x})\n')
        file.write('\n#endif // _ELT_PRELOAD_H_\n')
    print(f'Header file "{args.header_path}" generated successfully.')
