from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import numpy as np

world_size = 1
rank = 0
block_size = 128
# gemm_impl: Literal["bf16", "fp8"] = "bf16"
# attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "fp8"
    vocab_size: int = 129280
    dim: int = 7168
    # dim: int = 1024
    # dim: int = 2048
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    # moe_inter_dim: int = 512
    # moe_inter_dim: int = 1024
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    # moe
    # n_routed_experts: int = 256
    # n_routed_experts: int = 128
    n_routed_experts: int = 16
    # n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    # n_activated_experts: int = 4
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float = 2.5
    # mla
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

def linear(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
    # Convert to float32 for better precision during computation
    x_32 = x.astype(np.float32)
    weight_32 = weight.astype(np.float32)
    
    # Compute y = x @ weight.T
    y = np.matmul(x_32, weight_32.T)
    
    # Add bias if it exists
    if bias is not None:
        bias_32 = bias.astype(np.float32)
        y += bias_32
    
    # Convert back to float16 to match the original precision
    return y.astype(np.float16)

class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.normal(0, 0.01, (out_features, in_features)).astype(np.float16)
        self.bias = np.random.normal(0, 0.01, out_features).astype(np.float16)
        # can be None during inference
        # self.bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return linear(x, self.weight, self.bias).astype(np.float16)

class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        assert out_features % world_size == 0
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias)


class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        assert in_features % world_size == 0
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias)
    

class MLP:
    def __init__(self, dim: int, inter_dim: int):
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))

class Gate:
    def __init__(self, args: ModelArgs):
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = np.random.normal(0, 0.01, (args.n_routed_experts, args.dim)).astype(np.float16)
        # Disabled for simplicity
        # self.bias = np.random.normal(0, 0.02, args.n_routed_experts).astype(np.float16)
        self.bias = None

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scores = linear(x, self.weight)
        # if self.score_func == "softmax":
        #     scores = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        # else:
        #     scores = 1 / (1 + np.exp(-scores))

        # float32 used to match the precision of torch
        scores = scores.astype(np.float32)
        scores = 1. / (1. + np.exp(-scores))
        scores = scores.astype(np.float16)
        # print(scores)
        # print("=====")

        # copy required
        original_scores = scores.copy()
        # Auxiliary-Loss-Free Load Balancing
        # if self.bias is not None:
        #     scores += self.bias
        # # 
        # if self.n_groups > 1:
        #     scores = scores.reshape(x.shape[0], self.n_groups, -1)
        #     group_scores = np.amax(scores, axis=-1) if self.bias is None else np.sum(np.partition(scores, -2, axis=-1)[:, :, -2:], axis=-1)
        #     indices = np.argpartition(group_scores, -self.topk_groups, axis=-1)[:, -self.topk_groups:]
        #     mask = np.ones((x.shape[0], self.n_groups), dtype=bool)
        #     mask[np.arange(x.shape[0])[:, None], indices] = False
        #     scores = np.where(mask[:, :, None], float("-inf"), scores).reshape(x.shape[0], -1)
        # print("Original scores")
        # print(original_scores)
        # original_scores now match torch
        indices = np.argpartition(scores, -self.topk, axis=-1)[:, -self.topk:]
        
        # print(indices)
        weights = np.take_along_axis(original_scores, indices, axis=-1)
        # print(weights)
        if self.score_func == "sigmoid":
            # weights /= np.sum(weights, axis=-1, keepdims=True, dtype=np.float32)
            # use the np.divide for better precision control
            weights = np.divide(weights, np.sum(weights, axis=-1, keepdims=True, dtype=np.float32), dtype=np.float16)
        # print(weights)
        weights *= self.route_scale
        # print(weights)
        # print("=====")
        return weights.astype(x.dtype), indices

def silu(x: np.ndarray) -> np.ndarray:
    # x * sigmoid(x)
    y = x / (1. + np.exp(-x.astype(np.float32)))
    return y.astype(np.float16)

class Expert:
    def __init__(self, dim: int, inter_dim: int):
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        return self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))

class MoE:
    def __init__(self, args: ModelArgs):
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = [Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None for i in range(self.n_routed_experts)]
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        weights, indices = self.gate.forward(x)
        # print(weights)
        # print(indices)
        y = np.zeros_like(x)
        counts = np.bincount(indices.flatten(), minlength=self.n_routed_experts)
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = np.where(indices == i)
            # print(x[idx])
            # print(expert.forward(x[idx]))
            # print(weights[idx, top, None])
            y[idx] += expert.forward(x[idx]) * weights[idx, top, None]
        # print(y)
        z = self.shared_experts.forward(x)

        return (y + z).reshape(shape)
    
    def get_gate_weights(self) -> np.ndarray:
        return self.gate.weight.astype(np.float16)
    
    def get_expert_w1_weights(self) -> np.ndarray:
        return np.concatenate(([expert.w1.weight for expert in self.experts if expert is not None], self.shared_experts.w1.weight), axis=None).astype(np.float16)
    
    def get_expert_w1_bias(self) -> np.ndarray:
        return np.concatenate(([expert.w1.bias for expert in self.experts if expert is not None], self.shared_experts.w1.bias), axis=None).astype(np.float16)
    
    def get_expert_w2_weights(self) -> np.ndarray:
        return np.concatenate(([expert.w2.weight for expert in self.experts if expert is not None], self.shared_experts.w2.weight), axis=None).astype(np.float16)
    
    def get_expert_w2_bias(self) -> np.ndarray:
        return np.concatenate(([expert.w2.bias for expert in self.experts if expert is not None], self.shared_experts.w2.bias), axis=None).astype(np.float16)
    
    def get_expert_w3_weights(self) -> np.ndarray:
        return np.concatenate(([expert.w3.weight for expert in self.experts if expert is not None], self.shared_experts.w3.weight), axis=None).astype(np.float16)
    
    def get_expert_w3_bias(self) -> np.ndarray:
        return np.concatenate(([expert.w3.bias for expert in self.experts if expert is not None], self.shared_experts.w3.bias), axis=None).astype(np.float16)
    


if __name__ == "__main__":
    print("Golden Model with NumPy")
    np.set_printoptions(precision=6, floatmode="fixed")
    np.random.seed(1)
    args = ModelArgs()
    n_tokens = 3
    x = np.random.rand(n_tokens, args.dim).astype(np.float16)
    print("Input tensor:")
    print(x.shape)
    print(x)
    model = MoE(args)
    out = model.forward(x)
    print("Output tensor:")
    print(out.shape)
    print(out)