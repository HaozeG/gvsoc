import sys
import os

# Add the config directory to the Python module search path
config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.append(config_dir)

from arch_NoC512 import FlexClusterArch       # Read the hardware config for data placement
# from arch_hbm16_NoC512 import FlexClusterArch
import DeepseekMoE
import numpy as np
import preload as pld

DATA_SIZE_BYTES = 2

arch = FlexClusterArch()

# Function to split matrices into vertical tiles
def partition_matrices(matrices, n_matrices, width, rows, tile_width):
    # TODO: Handle the case where width is not divisible by tile_width
    assert width % tile_width == 0
    n_tiles = width // tile_width
    # matrix_size = width * rows
    
    matrices = matrices.reshape((n_matrices, rows, width))
    output = []
    
    for tile_id in range(n_tiles):
        tile_group = []
        start_col = tile_id * tile_width
        end_col = start_col + tile_width
        
        for m in range(n_matrices):
            tile = matrices[m][:, start_col:end_col]  # shape (rows, tile_width)
            tile_group.append(tile.flatten())  # flatten in row-major order
     
        # Combine all matrices for this tile
        tile_group_flat = np.concatenate(tile_group)   # shape (n_matrices * rows * tile_width,)
        output.append(tile_group_flat)
        
    return np.stack(output)  # shape: (n_tiles, n_matrices * rows * tile_width)


# Function to map expert weights and biases to HBM addresses
def map_w1w3_to_hbm(partitioned_weight_matrices, n_hbm_channels_utilized, hbm_base, hbm_node_addr_space, tile_size):
    weights_addresses = []
    bias_addresses = []
    
    for i in range(n_hbm_channels_utilized):
        tiles_per_channel = len(partitioned_weight_matrices) // n_hbm_channels_utilized
        for j in range(tiles_per_channel):
            weights_addresses.append(hbm_base + (i * hbm_node_addr_space) + (j * tile_size))
            if j == tiles_per_channel - 1:
                bias_addresses.append(hbm_base + (i * hbm_node_addr_space) + ((j + 1) * tile_size))

    return weights_addresses, bias_addresses


if __name__ == '__main__':
    np.random.seed(1)
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
    
    # print the shape of the weight matrices
    # print("expert_w1_weights: ", expert_w1_weights)
    # print("expert_w1_weights shape: ", expert_w1_weights.shape)
    # print("expert_w2_weights: ", expert_w2_weights)
    # print("expert_w2_weights shape: ", expert_w2_weights.shape)
    
    # Read the hardwre architecture configuration from the config file
    hbm_base_addr       = arch.hbm_start_base
    hbm_node_addr_space = arch.hbm_node_addr_space
    num_hbm_channels_w  = arch.hbm_chan_placement[0]     # West HBM
    num_hbm_channels_n  = arch.hbm_chan_placement[1]     # North HBM
    num_hbm_channels_e  = arch.hbm_chan_placement[2]     # South HBM
    num_hbm_channels_s  = arch.hbm_chan_placement[3]     # East HBM
    num_cluster_x       = arch.num_cluster_x
    num_cluster_y       = arch.num_cluster_y
    total_num_hbm_channels = num_hbm_channels_w + num_hbm_channels_n + num_hbm_channels_e + num_hbm_channels_s
    
    # Print the hardware architecture configuration
    # print("HBM base address: ", hex(hbm_base_addr))
    # print("HBM node address space: ", hex(hbm_node_addr_space))
    # print("Number of HBM channels (West): ", num_hbm_channels_w)
    # print("Number of HBM channels (North): ", num_hbm_channels_n)
    # print("Number of HBM channels (East): ", num_hbm_channels_e)
    # print("Number of HBM channels (South): ", num_hbm_channels_s)
    # print("Number of clusters in X direction: ", num_cluster_x)
    # print("Number of clusters in Y direction: ", num_cluster_y)
    
    # Base address maps of HBM channels
    # hbm_ch0_addr = hbm_base_addr
    # hbm_ch1_addr = hbm_base_addr + hbm_node_addr_space
    # hbm_ch2_addr = hbm_base_addr + hbm_node_addr_space * 2
    # hbm_ch3_addr = hbm_base_addr + hbm_node_addr_space * 3
    
    # HBM base maps
    # NOTE: Currently supports only W and S HBM channels, overhaul required for N and E channels
    hbm_west_base = hbm_base_addr
    hbm_north_base = hbm_base_addr + hbm_node_addr_space * num_cluster_y
    hbm_east_base = hbm_base_addr + hbm_node_addr_space * (num_cluster_y + num_cluster_x)
    hbm_south_base = hbm_base_addr + hbm_node_addr_space * (2 * num_cluster_y + num_cluster_x)
    
    # hbm_ch4_addr = hbm_south_base
    # hbm_ch5_addr = hbm_south_base + hbm_node_addr_space
    # hbm_ch6_addr = hbm_south_base + hbm_node_addr_space * 2
    # hbm_ch7_addr = hbm_south_base + hbm_node_addr_space * 3
    
    # Partition W1 and W3 matrices into vertical tiles
    tile_width_w1_w3 = moe_inter_dim // (num_hbm_channels_s + num_hbm_channels_w)
    n_total_experts = n_routed_experts + n_shared_experts
    expert_w1_weights_partitioned = partition_matrices(expert_w1_weights, n_total_experts, moe_inter_dim, dim, tile_width_w1_w3)
    expert_w3_weights_partitioned = partition_matrices(expert_w3_weights, n_total_experts, moe_inter_dim, dim, tile_width_w1_w3)
    # print("partitioned_expert_w1:", expert_w1_weights_partitioned)
    # print("partitioned_expert_w1 shape:", expert_w1_weights_partitioned.shape)
    
    # # Partition W2 matrix into vertical tiles
    tile_width_w2 = dim // (num_hbm_channels_s + num_hbm_channels_w)
    expert_w2_weights_partitioned = partition_matrices(expert_w2_weights, n_total_experts, dim, moe_inter_dim, tile_width_w2)
    # print("partitioned_expert_w2:", expert_w2_weights_partitioned)
    # print("partitioned_expert_w2 shape:", expert_w2_weights_partitioned.shape)
    
    # Map partitioned matrices to HBM channels
    # W1 accessed by coloring 0, store in channel 4, 5, 6, 7
    tile_size_w1_w3 = expert_w1_weights_partitioned.nbytes // len(expert_w1_weights_partitioned)
    expert_w1_weights_addresses = []
    expert_w1_bias_addresses = []
    expert_w1_weights_addresses, expert_w1_bias_addresses = map_w1w3_to_hbm(expert_w1_weights_partitioned, num_hbm_channels_s, hbm_south_base, hbm_node_addr_space, tile_size_w1_w3)
        
    print("expert_w1_weights_addresses: ", [hex(addr) for addr in expert_w1_weights_addresses])
    print("expert_w1_bias_addresses: ", [hex(addr) for addr in expert_w1_bias_addresses])
    
    # W3 accessed by coloring 1, store in channel 0, 1, 2, 3
    expert_w3_weigts_addresses = []
    expert_w3_bias_addresses = []                
    expert_w3_weigts_addresses, expert_w3_bias_addresses = map_w1w3_to_hbm(expert_w3_weights_partitioned, num_hbm_channels_w, hbm_west_base, hbm_node_addr_space, tile_size_w1_w3)
                
    print("expert_w3_weigts_addresses: ", [hex(addr) for addr in expert_w3_weigts_addresses])
    print("expert_w3_bias_addresses: ", [hex(addr) for addr in expert_w3_bias_addresses])
    
    # W2 accessed by all clusters
    tile_size_w2 = expert_w2_weights_partitioned.nbytes // len(expert_w2_weights_partitioned)
    expert_w2_weights_addresses = []
    expert_w2_bias_addresses = []
    for i in range(num_hbm_channels_w):
        expert_w2_weights_addresses.append(expert_w3_bias_addresses[i] + expert_w3_bias.nbytes)
        expert_w2_bias_addresses.append(expert_w2_weights_addresses[i] + tile_size_w2)
        
    for i in range(num_hbm_channels_s):
        j = i + num_hbm_channels_w
        expert_w2_weights_addresses.append(expert_w1_bias_addresses[i] + expert_w1_bias.nbytes)
        expert_w2_bias_addresses.append(expert_w2_weights_addresses[j] + tile_size_w2)
        
    print("expert_w2_weights_addresses: ", [hex(addr) for addr in expert_w2_weights_addresses])
    print("expert_w2_bias_addresses: ", [hex(addr) for addr in expert_w2_bias_addresses])

    # A copy of gate weights in all HBM channels
    gate_weights_addresses = []
    for i in range(num_hbm_channels_w + num_hbm_channels_s):
        gate_weights_addresses.append(expert_w2_bias_addresses[i] + experts_w2_bias.nbytes)
    
    print("gate_weights_addresses: ", [hex(addr) for addr in gate_weights_addresses])
    
    # All other data are placed only in HBM channel 0 following the gate weights
    in_token_address= gate_weights_addresses[0] + gate_weights.nbytes
    actual_out_address = in_token_address + in_token.nbytes
    golden_address = actual_out_address + actual_out.nbytes
    
    # Combine all addresses into a list for preloading
    pld_addresses = (
        expert_w1_weights_addresses +
        expert_w3_weigts_addresses +
        expert_w2_weights_addresses +
        gate_weights_addresses +
        expert_w1_bias_addresses +
        expert_w3_bias_addresses +
        expert_w2_bias_addresses +
        [in_token_address, actual_out_address, golden_address]
    )
    # print("pld_addresses: ", [hex(addr) for addr in pld_addresses])
    # print("ple_addresess shape: ", len(pld_addresses))
    
    # Combine all data into a list for preloading in same sequance as tge addresseses
    pld_data = []
    for i in range(len(expert_w1_weights_addresses)):
        pld_data.append(expert_w1_weights_partitioned[i])
    for i in range(len(expert_w3_weigts_addresses)):
        pld_data.append(expert_w3_weights_partitioned[i])
    for i in range(len(expert_w2_weights_addresses)):
        pld_data.append(expert_w2_weights_partitioned[i])
    for i in range(len(gate_weights_addresses)):
        pld_data.append(gate_weights)
    for i in range(len(expert_w1_bias_addresses)):
        pld_data.append(expert_w1_bias)
    for i in range(len(expert_w3_bias_addresses)):
        pld_data.append(expert_w3_bias)
    for i in range(len(expert_w2_bias_addresses)):
        pld_data.append(experts_w2_bias)
        
    # Append the trailing data
    pld_data.append(in_token)
    pld_data.append(actual_out)
    pld_data.append(golden)
    
    # Perform the preload
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pld.make_preload_elf("hbm_data_opt.elf", pld_data, pld_addresses)
    
    # FOR DEBUGGING PURPOSES ONLY
    # pld_data_ref =     [
    #                         expert_w1_weights_partitioned[0],
    #                         expert_w1_weights_partitioned[1],
    #                         expert_w1_weights_partitioned[2],
    #                         expert_w1_weights_partitioned[3],
    #                         expert_w1_weights_partitioned[4],
    #                         expert_w1_weights_partitioned[5],
    #                         expert_w1_weights_partitioned[6],
    #                         expert_w1_weights_partitioned[7],
                            
    #                         expert_w3_weights_partitioned[0],
    #                         expert_w3_weights_partitioned[1],
    #                         expert_w3_weights_partitioned[2],
    #                         expert_w3_weights_partitioned[3],
    #                         expert_w3_weights_partitioned[4],
    #                         expert_w3_weights_partitioned[5],
    #                         expert_w3_weights_partitioned[6],
    #                         expert_w3_weights_partitioned[7],
                            
    #                         expert_w2_weights_partitioned[0],
    #                         expert_w2_weights_partitioned[1],
    #                         expert_w2_weights_partitioned[2],
    #                         expert_w2_weights_partitioned[3],
    #                         expert_w2_weights_partitioned[4],
    #                         expert_w2_weights_partitioned[5],
    #                         expert_w2_weights_partitioned[6],
    #                         expert_w2_weights_partitioned[7],
                            
    #                         gate_weights,
    #                         gate_weights,
    #                         gate_weights,
    #                         gate_weights,
    #                         gate_weights,
    #                         gate_weights,
    #                         gate_weights,
    #                         gate_weights,
                            
    #                         expert_w1_bias,
    #                         expert_w1_bias,
    #                         expert_w1_bias,
    #                         expert_w1_bias,
                            
    #                         expert_w3_bias,
    #                         expert_w3_bias,
    #                         expert_w3_bias,
    #                         expert_w3_bias,
                            
    #                         experts_w2_bias,
    #                         experts_w2_bias,
    #                         experts_w2_bias,
    #                         experts_w2_bias,
    #                         experts_w2_bias,
    #                         experts_w2_bias,
    #                         experts_w2_bias,
    #                         experts_w2_bias,
                            
    #                         in_token,
    #                         actual_out,
    #                         golden
    #                         ]
    # print("Size matching: ", len(pld_data) == len(pld_data_ref))
    # if len(pld_data) != len(pld_data_ref):
    #     print("pld_data size: ", len(pld_data))
    #     print("pld_data_ref size: ", len(pld_data_ref))
        
    # for i in range(len(pld_data)):
    #     if pld_data[i].all() != pld_data_ref[i].all():
    #         print("Mismatch at index ", i)
    #         print("pld_data: ", pld_data[i])
    #         print("pld_data_ref: ", pld_data_ref[i])
    #         break
    # print("pld_data: ", pld_data)
    # print("pld_data_ref: ", pld_data_ref)


