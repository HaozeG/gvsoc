import DeepseekMoE

import os
import numpy as np
import preload as pld

if __name__ == '__main__':
    args = DeepseekMoE.ModelArgs()
    n_tokens = 1
    in_token = np.random.rand(n_tokens, args.dim).astype(np.float16)
    model = DeepseekMoE.MoE(args)

    # A_host = in_token
    n_tokens = np.uint16(n_tokens)
    dim = np.uint16(args.dim)
    moe_inter_dim = np.uint16(args.moe_inter_dim)
    n_routed_experts = np.uint16(args.n_routed_experts)
    n_shared_experts = np.uint16(args.n_shared_experts)
    n_activated_experts = np.uint16(args.n_activated_experts)
    gate_weights = model.get_gate_weights()
    expert_w1_weights = model.get_expert_w1_weights()
    expert_w1_bias = model.get_expert_w1_bias()
    expert_w2_weights = model.get_expert_w2_weights()
    experts_w2_bias = model.get_expert_w2_bias()
    expert_w3_weights = model.get_expert_w3_weights()
    expert_w3_bias = model.get_expert_w3_bias()
    actual_out = np.zeros((n_tokens, args.dim), dtype=np.float16)
    golden = model.forward(in_token).astype(np.float16)

    # check if nan exists
    if np.isnan(golden).any():
        print("nan exists in golden")
        exit(1)
    if np.isnan(expert_w1_weights).any():
        print("nan exists in expert_w1_weights")
        exit(1)
    if np.isnan(expert_w1_bias).any():
        print("nan exists in expert_w1_bias")
        exit(1)
    if np.isnan(expert_w2_weights).any():
        print("nan exists in expert_w2_weights")
        exit(1)
    if np.isnan(experts_w2_bias).any():
        print("nan exists in experts_w2_bias")
        exit(1)
    if np.isnan(expert_w3_weights).any():
        print("nan exists in expert_w3_weights")
        exit(1)
    if np.isnan(expert_w3_bias).any():
        print("nan exists in expert_w3_bias")
        exit(1)
    if np.isnan(gate_weights).any():
        print("nan exists in gate_weights")
        exit(1)

    # print the first 1 elements of each array
    print("in_token: ", in_token[0])
    print("gate_weights: ", gate_weights[0])
    print("expert_w1_weights: ", expert_w1_weights[0])
    print("expert_w1_bias: ", expert_w1_bias[0])
    print("expert_w2_weights: ", expert_w2_weights[0])
    print("experts_w2_bias: ", experts_w2_bias[0])
    print("expert_w3_weights: ", expert_w3_weights[0])
    print("expert_w3_bias: ", expert_w3_bias[0])
    print("actual_out: ", actual_out[0])
    print("golden: ", golden[0])
        
    
    hbm_base_address = 0xc0000000
    in_token_address = hbm_base_address
    # n_tokens_address = in_token_address + in_token.nbytes
    # dim_address = n_tokens_address + n_tokens.nbytes
    # moe_inter_dim_address = dim_address + dim.nbytes
    # n_routed_experts_address = moe_inter_dim_address + moe_inter_dim.nbytes
    # n_shared_experts_address = n_routed_experts_address + n_routed_experts.nbytes
    # n_activated_experts_address = n_shared_experts_address + n_shared_experts.nbytes
    # gate_weights_address = n_activated_experts_address + n_activated_experts.nbytes
    gate_weights_address = in_token_address + in_token.nbytes
    expert_w1_weights_address = gate_weights_address + gate_weights.nbytes
    expert_w1_bias_address = expert_w1_weights_address + expert_w1_weights.nbytes
    expert_w2_weights_address = expert_w1_bias_address + expert_w1_bias.nbytes
    expert_w2_bias_address = expert_w2_weights_address + expert_w2_weights.nbytes
    expert_w3_weights_address = expert_w2_bias_address + experts_w2_bias.nbytes
    expert_w3_bias_address = expert_w3_weights_address + expert_w3_weights.nbytes
    actual_out_address = expert_w3_bias_address + expert_w3_bias.nbytes
    golden_address = actual_out_address + actual_out.nbytes
    # create a uint32 np array to store the addresses
    # arg_in = np.array([in_token_address, n_tokens_address, dim_address, moe_inter_dim_address, n_routed_experts_address, n_shared_experts_address, n_activated_experts_address, gate_weights_address, expert_w1_weights_address, expert_w1_bias_address, expert_w2_weights_address, expert_w2_bias_address, expert_w3_weights_address, expert_w3_bias_address, actual_out_address, golden_address], dtype=np.uint32)
    addr = np.array([in_token_address, gate_weights_address, expert_w1_weights_address, expert_w1_bias_address, expert_w2_weights_address, expert_w2_bias_address, expert_w3_weights_address, expert_w3_bias_address, actual_out_address, golden_address], dtype=np.uint32)
    # print args in hex

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pld.make_preload_elf("hbm_data_MoE.elf", [in_token, gate_weights, expert_w1_weights, expert_w1_bias, expert_w2_weights, experts_w2_bias, expert_w3_weights, expert_w3_bias, actual_out, golden], addr)
