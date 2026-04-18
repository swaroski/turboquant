import numpy as np

class KVCodecFallback:
    def __init__(self, config):
        self.config = config

    def create_cache(self, batch_size, num_heads):
        return KVCacheStateFallback(self.config, batch_size, num_heads)

class KVCacheStateFallback:
    def __init__(self, config, batch_size, num_heads):
        self.config = config
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.keys = []
        self.values = []

    def append(self, keys, values):
        """
        keys/values: [batch, heads, seq, dim]
        """
        self.keys.append(keys)
        self.values.append(values)

    def attention_scores(self, query):
        """
        query: [batch, heads, 1, dim]
        """
        all_keys = np.concatenate(self.keys, axis=2) # [B, H, S, D]
        # [B, H, 1, D] @ [B, H, D, S] -> [B, H, 1, S]
        scores = np.matmul(query, all_keys.transpose(0, 1, 3, 2))
        return scores
