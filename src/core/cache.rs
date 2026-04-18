use crate::core::quantizer::{Quantizer, UniformQuantizer};
use crate::core::rotation::fwht_inplace;
use crate::core::error::Result;

pub struct KVCacheStateInternal {
    pub head_dim: usize,
    pub key_bits: usize,
    pub value_bits: usize,
    
    // Storage: [Batch, Head, Seq, PackedDim]
    pub packed_keys: Vec<u8>,
    pub key_scales: Vec<f32>,
    
    pub packed_values: Vec<u8>,
    pub value_scales: Vec<f32>,
    
    pub batch_size: usize,
    pub num_heads: usize,
    
    key_quantizer: UniformQuantizer,
    value_quantizer: UniformQuantizer,
}

impl KVCacheStateInternal {
    pub fn new(batch_size: usize, num_heads: usize, head_dim: usize, key_bits: usize, value_bits: usize) -> Self {
        Self {
            head_dim,
            key_bits,
            value_bits,
            packed_keys: Vec::new(),
            key_scales: Vec::new(),
            packed_values: Vec::new(),
            value_scales: Vec::new(),
            batch_size,
            num_heads,
            key_quantizer: UniformQuantizer::new(key_bits),
            value_quantizer: UniformQuantizer::new(value_bits),
        }
    }

    pub fn append(&mut self, keys: &[f32], values: &[f32], _b: usize, _h: usize, _s: usize) -> Result<()> {
        let key_packed_dim = self.head_dim * self.key_bits / 8;
        let val_packed_dim = self.head_dim * self.value_bits / 8;
        
        let mut k_tmp = keys.to_vec();
        fwht_inplace(&mut k_tmp)?;
        
        let mut k_packed = vec![0u8; key_packed_dim];
        let mut k_scale = 0.0f32;
        self.key_quantizer.quantize(&k_tmp, &mut k_packed, &mut k_scale)?;
        
        let mut v_packed = vec![0u8; val_packed_dim];
        let mut v_scale = 0.0f32;
        self.value_quantizer.quantize(values, &mut v_packed, &mut v_scale)?;
        
        self.packed_keys.extend(k_packed);
        self.key_scales.push(k_scale);
        self.packed_values.extend(v_packed);
        self.value_scales.push(v_scale);
        
        Ok(())
    }
    
    pub fn get_num_tokens(&self) -> usize {
        if self.batch_size == 0 || self.num_heads == 0 {
            0
        } else {
            self.key_scales.len() / (self.batch_size * self.num_heads)
        }
    }
}
