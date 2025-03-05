## Notes on files

`./analyze_trace_0.txt`
- executes `gemm(in_token_addr, gate_weights_addr, actual_out_addr, 64, n_token, n_routed_experts, zomem(0))` only.
- `in_token` are all $0.01$, `gate_weights` are all $-2$.
- Desired output is $-1.28$.
- Actual output is $-1.2578125$.
- PRINT_DEBUG is enabled.
- without PRINT_DEBUG, the `AVG BW` of controller0 is $83.63 \%$