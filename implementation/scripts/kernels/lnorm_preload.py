#
# Generate the LayerNorm preload: fp16 input, gamma, beta, an empty output and
# a golden, plus sw/LayerNorm/include/preload.h with their addresses.
#
# The golden replays the kernel's precision choices (see
# sw/LayerNorm/include/LayerNorm.h): fp32 statistics, fp16 mean m and scale r
# and residual m2 = fp16(mean - m) and scale r rounded to nearest even, then
# (((x - m) - m2) * r) * gamma + beta with one fp16 rounding per instruction. The script also prints how far that sequence is
# from a float64 LayerNorm.
#
import os
import argparse
import importlib.util
import numpy as np
import preload as pld


def import_module_from_path(module_path):
    """
    Dynamically import a module from an absolute path and mimic `from module import *`.
    """
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Cannot find a module at path: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(vars(module))
    return module


def lnorm_inputs(m, n, seed=0):
    """Rows with spread sigma in [0.1, 50] (log-spaced, cycled) and a nonzero
    mean of up to a few sigma plus an offset of up to 10."""
    rng = np.random.default_rng(seed)
    sigma = np.geomspace(0.1, 50.0, 32)[np.arange(m) % 32]
    mean = rng.uniform(-3.0, 3.0, m) * sigma + rng.uniform(-10.0, 10.0, m)
    x = mean[:, None] + sigma[:, None] * rng.normal(size=(m, n))
    gamma = rng.normal(1.0, 0.2, n)
    beta = rng.normal(0.0, 0.2, n)
    h = np.float16
    return x.astype(h), gamma.astype(h), beta.astype(h)


def fp32_ordered_sum(v):
    """Sum along axis 1 in fp32, one rounding per addition, in index order:
    what vfredusum.vs does in the GVSoC model. Valid while a row fits in one
    strip (n <= VLMAX at e16/m4, 4096 on the default arch), so that each fp32
    lane holds exactly one element before the reduction."""
    acc = np.zeros(v.shape[0], dtype=np.float32)
    for j in range(v.shape[1]):
        acc = (acc + v[:, j]).astype(np.float32)
    return acc


def lnorm_fp16_replay(x, gamma, beta, eps):
    h, f = np.float16, np.float32
    n = x.shape[1]
    xf = x.astype(f)                                          # vfwadd: exact widening
    mean = (fp32_ordered_sum(xf) / f(n)).astype(f)
    m = mean.astype(h)                                        # nearest even
    w = (xf - m.astype(f)[:, None]).astype(f)                 # vfwsub.vf
    s2 = fp32_ordered_sum((w * w).astype(f))                  # vfmacc on zero lanes, reduce
    delta = (mean - m.astype(f)).astype(f)
    s2n = (s2 / f(n)).astype(f)
    var = (s2n.astype(np.float64) - delta.astype(np.float64) ** 2).astype(f)   # fnmsub.s
    var = np.maximum(var, f(0))
    rstd = (f(1) / np.sqrt((var + f(eps)).astype(f))).astype(f)
    r = rstd.astype(h)
    m2 = delta.astype(h)
    with np.errstate(all="ignore"):
        y = (x - m[:, None]).astype(h)
        y = (y - m2[:, None]).astype(h)
        y = (y * r[:, None]).astype(h)
        y = (y * gamma[None, :]).astype(h)
        y = (y + beta[None, :]).astype(h)
    return y


def lnorm_reference(x, gamma, beta, eps):
    xd = x.astype(np.float64)
    mu = xd.mean(axis=1, keepdims=True)
    var = ((xd - mu) ** 2).mean(axis=1, keepdims=True)
    return (xd - mu) / np.sqrt(var + eps) * gamma.astype(np.float64) + beta.astype(np.float64)


def report_algorithm_error(x, gamma, beta, eps, golden):
    ref = lnorm_reference(x, gamma, beta, eps)
    g = golden.astype(np.float64)
    err = np.abs(g - ref)
    big = np.abs(ref) >= 0.1
    print(f"[lnorm] fp16 sequence vs float64 LayerNorm: max abs err {err.max():.2e}; "
          f"max rel err {np.max(err[big] / np.abs(ref[big])):.2e} where |ref| >= 0.1; "
          f"NaN outputs {int(np.isnan(g).sum())} of {g.size}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate the LayerNorm preload ELF and C header.")
    parser.add_argument("elf_path", type=str, help="Path of preload elf file")
    parser.add_argument("header_path", type=str, help="Path to LayerNorm preload C header file")
    parser.add_argument("module_paths", metavar="module_path", type=str, nargs="+",
                        help="Arch and kernel configuration files")
    args = parser.parse_args()

    for module_path in args.module_paths:
        absolute_path = os.path.abspath(module_path)
        module = import_module_from_path(absolute_path)
        print(f"Successfully imported: {module.__name__} from {absolute_path}")

    lnorm = LayerNorm()
    arch = FlexClusterArch()
    if lnorm.dtype != 'fp16':
        raise ValueError("LayerNorm supports fp16 only")

    x, gamma, beta = lnorm_inputs(lnorm.m_size, lnorm.n_size)
    golden = lnorm_fp16_replay(x, gamma, beta, lnorm.eps)
    report_algorithm_error(x, gamma, beta, lnorm.eps, golden)

    u16 = lambda a: a.view(np.uint16)
    I_np, G_np, B_np, O_golden = u16(x), u16(gamma), u16(beta), u16(golden)
    O_empty = np.zeros(I_np.shape, dtype=np.uint16)

    # all arrays below 4 GiB: the preload ELF is 32-bit
    I_addr = arch.hbm_start_base
    O_eaddr = I_addr + I_np.nbytes
    O_gaddr = O_eaddr + O_empty.nbytes
    G_addr = (O_gaddr + O_golden.nbytes + 63) & ~63
    B_addr = G_addr + G_np.nbytes
    B_addr = (B_addr + 63) & ~63
    assert B_addr + B_np.nbytes <= 1 << 32, "preload arrays must lie below 4 GiB (32-bit ELF)"
    for name, addr in (("I_addr", I_addr), ("O_eaddr", O_eaddr), ("O_gaddr", O_gaddr),
                       ("G_addr", G_addr), ("B_addr", B_addr)):
        print(f"{name} = {addr: #x}")

    pld.make_preload_elf(args.elf_path,
        [I_np,   O_empty, O_golden, G_np,   B_np],
        [I_addr, O_eaddr, O_gaddr,  G_addr, B_addr])

    with open(args.header_path, 'w') as file:
        file.write('#ifndef _LNORM_PRELOAD_H_\n')
        file.write('#define _LNORM_PRELOAD_H_\n\n')
        for name, addr in (("I_addr", I_addr), ("O_eaddr", O_eaddr), ("O_gaddr", O_gaddr),
                           ("G_addr", G_addr), ("B_addr", B_addr)):
            file.write(f'#define {name.upper()} ((uint64_t){addr: #x})\n')
        file.write('\n#endif // _LNORM_PRELOAD_H_\n')

    print(f'Header file "{args.header_path}" generated successfully.')
