use super::error::Result;

pub fn pack_4bit(unpacked: &[u8], packed: &mut [u8]) -> Result<()> {
    if unpacked.len() != packed.len() * 2 {
        return Err(super::error::TQError::Internal("Length mismatch in pack_4bit".into()));
    }
    
    for i in 0..packed.len() {
        let low = unpacked[i * 2] & 0x0F;
        let high = unpacked[i * 2 + 1] & 0x0F;
        packed[i] = low | (high << 4);
    }
    Ok(())
}

pub fn unpack_4bit(packed: &[u8], unpacked: &mut [u8]) -> Result<()> {
    if unpacked.len() != packed.len() * 2 {
        return Err(super::error::TQError::Internal("Length mismatch in unpack_4bit".into()));
    }
    
    for i in 0..packed.len() {
        unpacked[i * 2] = packed[i] & 0x0F;
        unpacked[i * 2 + 1] = (packed[i] >> 4) & 0x0F;
    }
    Ok(())
}
