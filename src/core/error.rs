use thiserror::Error;

#[derive(Error, Debug)]
pub enum TQError {
    #[error("Shape mismatch: expected {expected:?}, found {found:?}")]
    ShapeMismatch { expected: Vec<usize>, found: Vec<usize> },
    
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("Unsupported bit width: {0}")]
    UnsupportedBitWidth(usize),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, TQError>;
