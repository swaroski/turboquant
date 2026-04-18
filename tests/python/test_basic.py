import turboquant as tq
import numpy as np

def test_basic_append():
    config = tq.KVCodecConfig(head_dim=128, key_bits=4, value_bits=4)
    codec = tq.KVCodec(config)
    cache = codec.create_cache(batch_size=1, num_heads=32)
    
    keys = np.random.randn(1, 32, 1, 128).astype(np.float32)
    values = np.random.randn(1, 32, 1, 128).astype(np.float32)
    
    cache.append(keys, values)
    print(f"Tokens in cache: {cache.num_tokens}")
    assert cache.num_tokens == 1

if __name__ == "__main__":
    test_basic_append()
    print("Test passed!")
