import sys
import os

# Add the config directory to the Python module search path
config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.append(config_dir)

from arch_NoC512 import FlexClusterArch
import DeepseekMoE
import numpy as np
import preload as pld

DATA_SIZE_BYTES = 2

arch = FlexClusterArch()

# Function to partition matrices into vertical tiles
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
    
    # Print the hardware architecture configuration
    print("HBM base address: ", hex(hbm_base_addr))
    print("HBM node address space: ", hex(hbm_node_addr_space))
    print("Number of HBM channels (West): ", num_hbm_channels_w)
    print("Number of HBM channels (North): ", num_hbm_channels_n)
    print("Number of HBM channels (East): ", num_hbm_channels_e)
    print("Number of HBM channels (South): ", num_hbm_channels_s)
    print("Number of clusters in X direction: ", num_cluster_x)
    print("Number of clusters in Y direction: ", num_cluster_y)
    
    # Base address maps of HBM channels
    hbm_ch0_addr = hbm_base_addr
    hbm_ch1_addr = hbm_base_addr + hbm_node_addr_space
    hbm_ch2_addr = hbm_base_addr + hbm_node_addr_space * 2
    hbm_ch3_addr = hbm_base_addr + hbm_node_addr_space * 3
    
    hbm_south_base = hbm_base_addr + hbm_node_addr_space * (2 * num_cluster_y + num_cluster_x)
    
    hbm_ch4_addr = hbm_south_base
    hbm_ch5_addr = hbm_south_base + hbm_node_addr_space
    hbm_ch6_addr = hbm_south_base + hbm_node_addr_space * 2
    hbm_ch7_addr = hbm_south_base + hbm_node_addr_space * 3
    
    # Partition W1 and W3 matrices into vertical tiles
    tile_width_w1_w3 = moe_inter_dim // (num_hbm_channels_s + num_hbm_channels_w)
    n_total_experts = n_routed_experts + n_shared_experts
    expert_w1_weights_partitioned = partition_matrices(expert_w1_weights, n_total_experts, moe_inter_dim, dim, tile_width_w1_w3)
    expert_w3_weights_partitioned = partition_matrices(expert_w3_weights, n_total_experts, moe_inter_dim, dim, tile_width_w1_w3)
    # print("partitioned_expert_w1:", expert_w1_weights_partitioned)
    # print("partitioned_expert_w1 shape:", expert_w1_weights_partitioned.shape)
    
    # Partition W2 matrix into vertical tiles
    tile_width_w2 = dim // (num_hbm_channels_s + num_hbm_channels_w)
    expert_w2_weights_partitioned = partition_matrices(expert_w2_weights, n_total_experts, dim, moe_inter_dim, tile_width_w2)
    # print("partitioned_expert_w2:", expert_w2_weights_partitioned)
    # print("partitioned_expert_w2 shape:", expert_w2_weights_partitioned.shape)
    
    # Map partitioned matrices to HBM channels
    # W1 accessed by coloring 0, store in channel 4, 5, 6, 7
    # TODO: array version verified but not active when preloading
    tile_size_w1_w3 = expert_w1_weights_partitioned.nbytes // len(expert_w1_weights_partitioned)
    expert_w1_weights_addresses = []
    expert_w1_bias_addresses = []
    for i in range(num_hbm_channels_s):
        tiles_per_channel = len(expert_w1_weights_partitioned) // num_hbm_channels_s
        for j in range(tiles_per_channel):
            expert_w1_weights_addresses.append(hbm_south_base + (i * hbm_node_addr_space) + (j * tile_size_w1_w3))
            if j == tiles_per_channel - 1:
                expert_w1_bias_addresses.append(hbm_south_base + (i * hbm_node_addr_space) + ((j + 1) * tile_size_w1_w3))
        
    # print("expert_w1_weights_addresses: ", [hex(addr) for addr in expert_w1_weights_addresses])
    # print("expert_w1_bias_addresses: ", [hex(addr) for addr in expert_w1_bias_addresses])
    
    expert_w1_weights_address_4_1 = hbm_ch4_addr
    expert_w1_weights_address_4_2 = hbm_ch4_addr + tile_size_w1_w3
    expert_w1_weights_address_5_1 = hbm_ch5_addr
    expert_w1_weights_address_5_2 = hbm_ch5_addr + tile_size_w1_w3
    expert_w1_weights_address_6_1 = hbm_ch6_addr
    expert_w1_weights_address_6_2 = hbm_ch6_addr + tile_size_w1_w3
    expert_w1_weights_address_7_1 = hbm_ch7_addr
    expert_w1_weights_address_7_2 = hbm_ch7_addr + tile_size_w1_w3
    expert_w1_bias_address_4 = expert_w1_weights_address_4_2 + tile_size_w1_w3
    expert_w1_bias_address_5 = expert_w1_weights_address_5_2 + tile_size_w1_w3
    expert_w1_bias_address_6 = expert_w1_weights_address_6_2 + tile_size_w1_w3
    expert_w1_bias_address_7 = expert_w1_weights_address_7_2 + tile_size_w1_w3
    
    # W3 accessed by coloring 1, store in channel 0, 1, 2, 3
    # TODO: array version verified but not active when preloading
    expert_w3_matrix_addresses = []
    expert_w3_bias_addresses = []
    for i in range(num_hbm_channels_w):
        tiles_per_channel = len(expert_w3_weights_partitioned) // num_hbm_channels_w
        for j in range(tiles_per_channel):
            expert_w3_matrix_addresses.append(hbm_base_addr + (i * hbm_node_addr_space) + (j * tile_size_w1_w3))
            if j == tiles_per_channel - 1:
                expert_w3_bias_addresses.append(hbm_base_addr + (i * hbm_node_addr_space) + ((j + 1) * tile_size_w1_w3))
                
    # print("expert_w3_matrix_addresses: ", [hex(addr) for addr in expert_w3_matrix_addresses])
    # print("expert_w3_bias_addresses: ", [hex(addr) for addr in expert_w3_bias_addresses])
    
    expert_w3_weights_address_0_1 = hbm_ch0_addr
    expert_w3_weights_address_0_2 = hbm_ch0_addr + tile_size_w1_w3
    expert_w3_weights_address_1_1 = hbm_ch1_addr
    expert_w3_weights_address_1_2 = hbm_ch1_addr + tile_size_w1_w3
    expert_w3_weights_address_2_1 = hbm_ch2_addr
    expert_w3_weights_address_2_2 = hbm_ch2_addr + tile_size_w1_w3
    expert_w3_weights_address_3_1 = hbm_ch3_addr
    expert_w3_weights_address_3_2 = hbm_ch3_addr + tile_size_w1_w3
    expert_w3_bias_address_0 = expert_w3_weights_address_0_2 + tile_size_w1_w3
    expert_w3_bias_address_1 = expert_w3_weights_address_1_2 + tile_size_w1_w3
    expert_w3_bias_address_2 = expert_w3_weights_address_2_2 + tile_size_w1_w3
    expert_w3_bias_address_3 = expert_w3_weights_address_3_2 + tile_size_w1_w3
    
    # W2 accessed by all clusters
    # TODO: array version verified but not active when preloading
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
        
    # print("expert_w2_weights_addresses: ", [hex(addr) for addr in expert_w2_weights_addresses])
    # print("expert_w2_bias_addresses: ", [hex(addr) for addr in expert_w2_bias_addresses])
    
    expert_w2_weights_address_0 = expert_w3_bias_address_0 + expert_w3_bias.nbytes
    epxert_w2_weights_address_1 = expert_w3_bias_address_1 + expert_w3_bias.nbytes
    expert_w2_weights_address_2 = expert_w3_bias_address_2 + expert_w3_bias.nbytes
    expert_w2_weights_address_3 = expert_w3_bias_address_3 + expert_w3_bias.nbytes
    expert_w2_weights_address_4 = expert_w1_bias_address_4 + expert_w1_bias.nbytes
    expert_w2_weights_address_5 = expert_w1_bias_address_5 + expert_w1_bias.nbytes
    expert_w2_weights_address_6 = expert_w1_bias_address_6 + expert_w1_bias.nbytes
    expert_w2_weights_address_7 = expert_w1_bias_address_7 + expert_w1_bias.nbytes
    expert_w2_bias_address_0 = expert_w2_weights_address_0 + tile_size_w2
    expert_w2_bias_address_1 = epxert_w2_weights_address_1 + tile_size_w2
    expert_w2_bias_address_2 = expert_w2_weights_address_2 + tile_size_w2
    expert_w2_bias_address_3 = expert_w2_weights_address_3 + tile_size_w2
    expert_w2_bias_address_4 = expert_w2_weights_address_4 + tile_size_w2
    expert_w2_bias_address_5 = expert_w2_weights_address_5 + tile_size_w2
    expert_w2_bias_address_6 = expert_w2_weights_address_6 + tile_size_w2
    expert_w2_bias_address_7 = expert_w2_weights_address_7 + tile_size_w2

    # A copy of gate weights in all HBM channels
    # TODO: array version verified but not active when preloading
    gate_weights_addresses = []
    for i in range(num_hbm_channels_w + num_hbm_channels_s):
        gate_weights_addresses.append(expert_w2_bias_addresses[i] + experts_w2_bias.nbytes)
    
    # print("gate_weights_addresses: ", [hex(addr) for addr in gate_weights_addresses])
    
    gate_weights_address_0 = expert_w2_bias_address_0 + experts_w2_bias.nbytes
    gate_weights_address_1 = expert_w2_bias_address_1 + experts_w2_bias.nbytes
    gate_weights_address_2 = expert_w2_bias_address_2 + experts_w2_bias.nbytes
    gate_weights_address_3 = expert_w2_bias_address_3 + experts_w2_bias.nbytes
    gate_weights_address_4 = expert_w2_bias_address_4 + experts_w2_bias.nbytes
    gate_weights_address_5 = expert_w2_bias_address_5 + experts_w2_bias.nbytes
    gate_weights_address_6 = expert_w2_bias_address_6 + experts_w2_bias.nbytes
    gate_weights_address_7 = expert_w2_bias_address_7 + experts_w2_bias.nbytes
    
    # All other data are placed only in HBM channel 0 after the gate weights
    # TODO: new version verified but not active
    in_token_address_new = gate_weights_addresses[0] + gate_weights.nbytes
    actual_out_address_new = in_token_address_new + in_token.nbytes
    golden_address_new = actual_out_address_new + actual_out.nbytes
    
    in_token_address = gate_weights_address_0 + gate_weights.nbytes
    actual_out_address = in_token_address + in_token.nbytes
    golden_address = actual_out_address + actual_out.nbytes
    
    
    ### Redundant version (backup reference) ###
    # W1 accessed by coloring 0, store in channel 0, 1, 2, 3
    # expert_w1_weights_address_4 = hbm_ch4_addr
    # expert_w1_weights_address_5 = hbm_ch5_addr
    # expert_w1_weights_address_6 = hbm_ch6_addr
    # expert_w1_weights_address_7 = hbm_ch7_addr
    # expert_w1_bias_address_4 = expert_w1_weights_address_4 + expert_w1_weights.nbytes
    # expert_w1_bias_address_5 = expert_w1_weights_address_5 + expert_w1_weights.nbytes
    # expert_w1_bias_address_6 = expert_w1_weights_address_6 + expert_w1_weights.nbytes
    # expert_w1_bias_address_7 = expert_w1_weights_address_7 + expert_w1_weights.nbytes
    
    # # W3 accessed by coloring 1, store in channel 4, 5, 6, 7
    # expert_w3_weights_address_0 = hbm_ch0_addr
    # expert_w3_weights_address_1 = hbm_ch1_addr
    # expert_w3_weights_address_2 = hbm_ch2_addr
    # expert_w3_weights_address_3 = hbm_ch3_addr
    # expert_w3_bias_address_0 = expert_w3_weights_address_0 + expert_w3_weights.nbytes
    # expert_w3_bias_address_1 = expert_w3_weights_address_1 + expert_w3_weights.nbytes
    # expert_w3_bias_address_2 = expert_w3_weights_address_2 + expert_w3_weights.nbytes
    # expert_w3_bias_address_3 = expert_w3_weights_address_3 + expert_w3_weights.nbytes
    
    # # W2 accessed by all clusters, store a copy in all channels    
    # expert_w2_weights_address_0 = expert_w3_bias_address_0 + expert_w3_bias.nbytes
    # expert_w2_weights_address_1 = expert_w3_bias_address_1 + expert_w3_bias.nbytes
    # expert_w2_weights_address_2 = expert_w3_bias_address_2 + expert_w3_bias.nbytes
    # expert_w2_weights_address_3 = expert_w3_bias_address_3 + expert_w3_bias.nbytes
    # expert_w2_weights_address_4 = expert_w1_bias_address_4 + expert_w1_bias.nbytes
    # expert_w2_weights_address_5 = expert_w1_bias_address_5 + expert_w1_bias.nbytes
    # expert_w2_weights_address_6 = expert_w1_bias_address_6 + expert_w1_bias.nbytes
    # expert_w2_weights_address_7 = expert_w1_bias_address_7 + expert_w1_bias.nbytes
    # expert_w2_bias_address_0 = expert_w2_weights_address_0 + expert_w2_weights.nbytes
    # expert_w2_bias_address_1 = expert_w2_weights_address_1 + expert_w2_weights.nbytes
    # expert_w2_bias_address_2 = expert_w2_weights_address_2 + expert_w2_weights.nbytes
    # expert_w2_bias_address_3 = expert_w2_weights_address_3 + expert_w2_weights.nbytes
    # expert_w2_bias_address_4 = expert_w2_weights_address_4 + expert_w2_weights.nbytes
    # expert_w2_bias_address_5 = expert_w2_weights_address_5 + expert_w2_weights.nbytes
    # expert_w2_bias_address_6 = expert_w2_weights_address_6 + expert_w2_weights.nbytes
    # expert_w2_bias_address_7 = expert_w2_weights_address_7 + expert_w2_weights.nbytes
    
    # # A copy of gate weights in all HBM channels
    # gate_weights_address_0 = expert_w2_bias_address_0 + experts_w2_bias.nbytes
    # gate_weights_address_1 = expert_w2_bias_address_1 + experts_w2_bias.nbytes
    # gate_weights_address_2 = expert_w2_bias_address_2 + experts_w2_bias.nbytes
    # gate_weights_address_3 = expert_w2_bias_address_3 + experts_w2_bias.nbytes
    # gate_weights_address_4 = expert_w2_bias_address_4 + experts_w2_bias.nbytes
    # gate_weights_address_5 = expert_w2_bias_address_5 + experts_w2_bias.nbytes
    # gate_weights_address_6 = expert_w2_bias_address_6 + experts_w2_bias.nbytes
    # gate_weights_address_7 = expert_w2_bias_address_7 + experts_w2_bias.nbytes
    
    # # All other data are placed only in HBM channel 0 after the gate weights
    # in_token_address = gate_weights_address_0 + gate_weights.nbytes
    # actual_out_address = in_token_address + in_token.nbytes
    # golden_address = actual_out_address + actual_out.nbytes
    ### END redundant verison ###
        
        
    # Print all addresses
    # print("in_token_address: ", hex(in_token_address))
    # print("gate_weights_address: ", hex(gate_weights_address))
    # print("expert_w1_weights_address: ", hex(expert_w1_weights_address))
    # print("expert_w2_weights_address: ", hex(expert_w2_weights_address))
    # print("expert_w3_weights_address: ", hex(expert_w3_weights_address))
    # print("expert_w1_bias_address: ", hex(expert_w1_bias_address))
    # print("expert_w2_bias_address: ", hex(expert_w2_bias_address))
    # print("expert_w3_bias_address: ", hex(expert_w3_bias_address))
    # print("actual_out_address: ", hex(actual_out_address))
    # print("golden_address: ", hex(golden_address))
        
    # Print data sizes
    # print("token size: ", in_token.nbytes)
    # print("gate_weights size: ", gate_weights.nbytes)
    # print("expert_w1_weights size: ", expert_w1_weights.nbytes)
    # print("expert_w1_bias size: ", expert_w1_bias.nbytes)
    # print("expert_w2_weights size: ", expert_w2_weights.nbytes)
    # print("experts_w2_bias size: ", experts_w2_bias.nbytes)
    # print("expert_w3_weights size: ", expert_w3_weights.nbytes)
    # print("expert_w3_bias size: ", expert_w3_bias.nbytes)
    # print("actual_out size: ", actual_out.nbytes)
    # print("golden size: ", golden.nbytes)
    # print("total size: ", in_token.nbytes + gate_weights.nbytes + expert_w1_weights.nbytes + expert_w1_bias.nbytes + expert_w2_weights.nbytes + experts_w2_bias.nbytes + expert_w3_weights.nbytes + expert_w3_bias.nbytes + actual_out.nbytes + golden.nbytes)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # TODO: use the array version for automated preloading
    pld.make_preload_elf("hbm_data_opt.elf", 
                            [expert_w1_weights_partitioned[0],
                            expert_w1_weights_partitioned[1],
                            expert_w1_weights_partitioned[2],
                            expert_w1_weights_partitioned[3],
                            expert_w1_weights_partitioned[4],
                            expert_w1_weights_partitioned[5],
                            expert_w1_weights_partitioned[6],
                            expert_w1_weights_partitioned[7],
                            expert_w1_bias,
                            expert_w1_bias,
                            expert_w1_bias,
                            expert_w1_bias,
                            
                            expert_w3_weights_partitioned[0],
                            expert_w3_weights_partitioned[1],
                            expert_w3_weights_partitioned[2],
                            expert_w3_weights_partitioned[3],
                            expert_w3_weights_partitioned[4],
                            expert_w3_weights_partitioned[5],
                            expert_w3_weights_partitioned[6],
                            expert_w3_weights_partitioned[7],
                            expert_w3_bias,
                            expert_w3_bias,
                            expert_w3_bias,
                            expert_w3_bias,
                            
                            expert_w2_weights_partitioned[0],
                            expert_w2_weights_partitioned[1],
                            expert_w2_weights_partitioned[2],
                            expert_w2_weights_partitioned[3],
                            expert_w2_weights_partitioned[4],
                            expert_w2_weights_partitioned[5],
                            expert_w2_weights_partitioned[6],
                            expert_w2_weights_partitioned[7],
                            experts_w2_bias,
                            experts_w2_bias,
                            experts_w2_bias,
                            experts_w2_bias,
                            experts_w2_bias,
                            experts_w2_bias,
                            experts_w2_bias,
                            experts_w2_bias,
                            
                            gate_weights,
                            gate_weights,
                            gate_weights,
                            gate_weights,
                            gate_weights,
                            gate_weights,
                            gate_weights,
                            gate_weights,
                            
                            in_token,
                            actual_out,
                            golden],
                            [expert_w1_weights_address_4_1,
                            expert_w1_weights_address_4_2,
                            expert_w1_weights_address_5_1,
                            expert_w1_weights_address_5_2,
                            expert_w1_weights_address_6_1,
                            expert_w1_weights_address_6_2,
                            expert_w1_weights_address_7_1,
                            expert_w1_weights_address_7_2,
                            expert_w1_bias_address_4,
                            expert_w1_bias_address_5,
                            expert_w1_bias_address_6,
                            expert_w1_bias_address_7,
                            
                            expert_w3_weights_address_0_1,
                            expert_w3_weights_address_0_2,
                            expert_w3_weights_address_1_1,
                            expert_w3_weights_address_1_2,
                            expert_w3_weights_address_2_1,
                            expert_w3_weights_address_2_2,
                            expert_w3_weights_address_3_1,
                            expert_w3_weights_address_3_2,
                            expert_w3_bias_address_0,
                            expert_w3_bias_address_1,
                            expert_w3_bias_address_2,
                            expert_w3_bias_address_3,
                            
                            expert_w2_weights_address_0,
                            epxert_w2_weights_address_1,
                            expert_w2_weights_address_2,
                            expert_w2_weights_address_3,
                            expert_w2_weights_address_4,
                            expert_w2_weights_address_5,
                            expert_w2_weights_address_6,
                            expert_w2_weights_address_7,
                            expert_w2_bias_address_0,
                            expert_w2_bias_address_1,
                            expert_w2_bias_address_2,
                            expert_w2_bias_address_3,
                            expert_w2_bias_address_4,
                            expert_w2_bias_address_5,
                            expert_w2_bias_address_6,
                            expert_w2_bias_address_7,
                            
                            gate_weights_address_0,
                            gate_weights_address_1,
                            gate_weights_address_2,
                            gate_weights_address_3,
                            gate_weights_address_4,
                            gate_weights_address_5,
                            gate_weights_address_6,
                            gate_weights_address_7,
                            
                            in_token_address,
                            actual_out_address,
                            golden_address])

    # Version 4
    # pld.make_preload_elf("hbm_data_decode.elf", 
    #                         [expert_w1_weights, 
    #                         expert_w1_weights, 
    #                         expert_w1_weights, 
    #                         expert_w1_weights, 
    #                         expert_w1_bias, 
    #                         expert_w1_bias, 
    #                         expert_w1_bias, 
    #                         expert_w1_bias, 
                            
    #                         expert_w3_weights, 
    #                         expert_w3_weights, 
    #                         expert_w3_weights, 
    #                         expert_w3_weights, 
    #                         expert_w3_bias, 
    #                         expert_w3_bias, 
    #                         expert_w3_bias, 
    #                         expert_w3_bias, 
                            
    #                         expert_w2_weights, 
    #                         expert_w2_weights, 
    #                         expert_w2_weights, 
    #                         expert_w2_weights, 
    #                         expert_w2_weights, 
    #                         expert_w2_weights, 
    #                         expert_w2_weights, 
    #                         expert_w2_weights, 
    #                         experts_w2_bias, 
    #                         experts_w2_bias, 
    #                         experts_w2_bias, 
    #                         experts_w2_bias, 
    #                         experts_w2_bias, 
    #                         experts_w2_bias, 
    #                         experts_w2_bias, 
    #                         experts_w2_bias, 
                            
    #                         gate_weights,
    #                         gate_weights,
    #                         gate_weights,
    #                         gate_weights,
    #                         gate_weights,
    #                         gate_weights,
    #                         gate_weights,
    #                         gate_weights,
                            
    #                         in_token,
    #                         actual_out, 
    #                         golden], 
    #                         [expert_w1_weights_address_4,
    #                         expert_w1_weights_address_5,
    #                         expert_w1_weights_address_6,
    #                         expert_w1_weights_address_7,
    #                         expert_w1_bias_address_4,
    #                         expert_w1_bias_address_5,
    #                         expert_w1_bias_address_6,
    #                         expert_w1_bias_address_7,
                            
    #                         expert_w3_weights_address_0,
    #                         expert_w3_weights_address_1,
    #                         expert_w3_weights_address_2,
    #                         expert_w3_weights_address_3,
    #                         expert_w3_bias_address_0,
    #                         expert_w3_bias_address_1,
    #                         expert_w3_bias_address_2,
    #                         expert_w3_bias_address_3,
                            
    #                         expert_w2_weights_address_0,
    #                         expert_w2_weights_address_1,
    #                         expert_w2_weights_address_2,
    #                         expert_w2_weights_address_3,
    #                         expert_w2_weights_address_4,
    #                         expert_w2_weights_address_5,
    #                         expert_w2_weights_address_6,
    #                         expert_w2_weights_address_7,
    #                         expert_w2_bias_address_0,
    #                         expert_w2_bias_address_1,
    #                         expert_w2_bias_address_2,
    #                         expert_w2_bias_address_3,
    #                         expert_w2_bias_address_4,
    #                         expert_w2_bias_address_5,
    #                         expert_w2_bias_address_6,
    #                         expert_w2_bias_address_7,
                            
    #                         gate_weights_address_0,
    #                         gate_weights_address_1,
    #                         gate_weights_address_2,
    #                         gate_weights_address_3,
    #                         gate_weights_address_4,
    #                         gate_weights_address_5,
    #                         gate_weights_address_6,
    #                         gate_weights_address_7,
                            
    #                         in_token_address,
    #                         actual_out_address,
    #                         golden_address
    #                       ])


