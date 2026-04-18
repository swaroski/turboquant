import turboquant as tq
import numpy as np
import torch
from typing import Optional

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
        return tensor.detach().cpu().to(torch.float32).numpy()
    return tensor.detach().cpu().numpy()

class TorchKVCache:
    def __init__(self, config: tq.KVCodecConfig, batch_size: int, num_heads: int, device: str = "cpu"):
        self.codec = tq.KVCodec(config)
        self.cache = self.codec.create_cache(batch_size, num_heads)
        self.device = device

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        """
        keys/values: [B, H, S, D]
        """
        k_np = to_numpy(keys)
        v_np = to_numpy(values)
        self.cache.append(k_np, v_np)

    def attention_scores(self, query: torch.Tensor) -> torch.Tensor:
        """
        query: [B, H, 1, D]
        """
        q_np = to_numpy(query)
        scores_np = self.cache.attention_scores(q_np)
        return torch.from_numpy(scores_np).to(self.device)
