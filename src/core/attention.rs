use crate::core::error::Result;
use crate::core::cache::KVCacheStateInternal;
use crate::core::quantizer::{Quantizer, UniformQuantizer};

pub fn attention_scores_packed(
    query: &[f32], // [Batch, Head, 1, Dim]
    cache: &KVCacheStateInternal,
) -> Result<Vec<f32>> {
    let batch_size = cache.batch_size;
    let num_heads = cache.num_heads;
    let head_dim = cache.head_dim;
    let num_tokens = cache.get_num_tokens();
    
    let mut scores = vec![0.0f32; batch_size * num_heads * num_tokens];
    
    let key_packed_dim = head_dim * cache.key_bits / 8;
    let quantizer = UniformQuantizer::new(cache.key_bits);
    
    // For now, naive loop. In Phase 3 we optimize with SIMD.
    for b in 0..batch_size {
        for h in 0..num_heads {
            let q_offset = (b * num_heads + h) * head_dim;
            let q_vec = &query[q_offset..q_offset + head_dim];
            
            for s in 0..num_tokens {
                let cache_idx = (b * num_heads * num_tokens) + (h * num_tokens) + s;
                let k_packed_offset = cache_idx * key_packed_dim;
                let k_packed = &cache.packed_keys[k_packed_offset..k_packed_offset + key_packed_dim];
                let k_scale = cache.key_scales[cache_idx];
                
                let mut k_unpacked = vec![0.0f32; head_dim];
                quantizer.dequantize(k_packed, k_scale, &mut k_unpacked)?;
                
                // Dot product
                let mut sum = 0.0f32;
                for i in 0..head_dim {
                    sum += q_vec[i] * k_unpacked[i];
                }
                
                scores[(b * num_heads + h) * num_tokens + s] = sum;
            }
        }
    }
    
    Ok(scores)
}
