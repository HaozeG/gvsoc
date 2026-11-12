# Kernel for End-to-End Dataflow Implementation for Modern LLMs

## 🚀 Getting Started with Kernel Execution

### ⚙️ Supported Kernels and Short Names

| Kernel Description                        | Short Name |
| ----------------------------------------- | ---------- |
| 🪞 FlatAttention (MHA, MQA, GQA)          | `attn`     |
| 🔢 GEMM with SUMMA Dataflow               | `gemm`     |
| 🌈 RMSNorm                                | `norm`     |
| 📏 LayerNorm (fp16, fp32 statistics)      | `lnorm`    |
| ✨ Activation (Sigmoid, ReLU, SiLU, GELU) | `acti`     |
| ➕ Eltwise (add, bias, scale, axpy, RoPE, pixel shuffle) | `elt` |

Activation, LayerNorm and Eltwise check their output on the device: the preload holds
a golden made by replaying the kernel's fp16 instruction sequence in numpy,
and after the timed region each cluster prints
`[check] cluster <id>: rows .., bad .., max_ulp ..` (a nonzero `bad` is the
EOC value). The preload script also prints the sequence's error against a
float64 reference. SmolVLA vision-encoder shapes:
`make acti-run acti=config/kernels/acti_gelu_siglip.py`, `make lnorm-run`.
Eltwise picks its operator with `op` in `config/kernels/elt.py`: `make elt-run`.


---

### 🏗️ Build, Preload, and Run Simulation

To build configuration, prepare preload data, and run simulation, use:

```bash
make <Kernel>-run
```
The defualt kernel configuration are located in `config/kernels`
