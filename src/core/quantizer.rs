use crate::core::error::Result;

pub trait Quantizer {
    fn quantize(&self, data: &[f32], packed: &mut [u8], scale: &mut f32) -> Result<()>;
    fn dequantize(&self, packed: &[u8], scale: f32, data: &mut [f32]) -> Result<()>;
}

pub struct UniformQuantizer {
    bits: usize,
}

impl UniformQuantizer {
    pub fn new(bits: usize) -> Self {
        Self { bits }
    }
}

impl Quantizer for UniformQuantizer {
    fn quantize(&self, data: &[f32], packed: &mut [u8], scale: &mut f32) -> Result<()> {
        if self.bits != 4 {
            return Err(crate::core::error::TQError::UnsupportedBitWidth(self.bits));
        }
        
        let mut max_abs = 0.0f32;
        for &x in data {
            max_abs = max_abs.max(x.abs());
        }
        
        *scale = max_abs / 7.0; // 4-bit signed range is -8..7, using 7 to avoid overflow
        
        let mut unpacked = vec![0u8; data.len()];
        for i in 0..data.len() {
            let q = if *scale > 0.0 {
                (data[i] / *scale).round().clamp(-8.0, 7.0) as i8
            } else {
                0
            };
            unpacked[i] = (q + 8) as u8; // Shift to 0..15
        }
        
        super::packing::pack_4bit(&unpacked, packed)?;
        Ok(())
    }

    fn dequantize(&self, packed: &[u8], scale: f32, data: &mut [f32]) -> Result<()> {
        if self.bits != 4 {
            return Err(crate::core::error::TQError::UnsupportedBitWidth(self.bits));
        }
        
        let mut unpacked = vec![0u8; data.len()];
        super::packing::unpack_4bit(packed, &mut unpacked)?;
        
        for i in 0..data.len() {
            let q = (unpacked[i] as i16 - 8) as f32;
            data[i] = q * scale;
        }
        Ok(())
    }
}
