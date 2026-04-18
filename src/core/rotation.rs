use crate::core::error::Result;

pub fn fwht_inplace(data: &mut [f32]) -> Result<()> {
    let n = data.len();
    if !n.is_power_of_two() {
        return Err(crate::core::error::TQError::InvalidConfig("FWHT requires power of two length".into()));
    }
    
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }
    
    // Normalize
    let norm = (n as f32).sqrt();
    for x in data.iter_mut() {
        *x /= norm;
    }
    
    Ok(())
}
