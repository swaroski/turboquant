use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use numpy::{PyArray1, PyArray2, PyArray4, PyReadonlyArray4, ToPyArray};
use std::sync::{Arc, Mutex};

mod core;

use crate::core::error::TQError;
use crate::core::cache::KVCacheStateInternal;

impl From<TQError> for PyErr {
    fn from(err: TQError) -> PyErr {
        match err {
            TQError::ShapeMismatch { .. } => PyValueError::new_err(err.to_string()),
            TQError::InvalidConfig(_) => PyValueError::new_err(err.to_string()),
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct KVCodecConfig {
    #[pyo3(get, set)]
    pub head_dim: usize,
    #[pyo3(get, set)]
    pub key_bits: usize,
    #[pyo3(get, set)]
    pub value_bits: usize,
    #[pyo3(get, set)]
    pub residual_window: usize,
}

#[pymethods]
impl KVCodecConfig {
    #[new]
    #[pyo3(signature = (head_dim=128, key_bits=4, value_bits=4, residual_window=32))]
    fn new(head_dim: usize, key_bits: usize, value_bits: usize, residual_window: usize) -> Self {
        KVCodecConfig {
            head_dim,
            key_bits,
            value_bits,
            residual_window,
        }
    }
}

#[pyclass]
pub struct KVCodec {
    config: KVCodecConfig,
}

#[pymethods]
impl KVCodec {
    #[new]
    fn new(config: KVCodecConfig) -> Self {
        KVCodec { config }
    }

    fn create_cache(&self, batch_size: usize, num_heads: usize) -> KVCacheState {
        KVCacheState::new(self.config.clone(), batch_size, num_heads)
    }
}

#[pyclass]
pub struct KVCacheState {
    inner: Arc<Mutex<KVCacheStateInternal>>,
    config: KVCodecConfig,
    batch_size: usize,
    num_heads: usize,
}

#[pymethods]
impl KVCacheState {
    #[new]
    fn new(config: KVCodecConfig, batch_size: usize, num_heads: usize) -> Self {
        let inner = KVCacheStateInternal::new(
            batch_size,
            num_heads,
            config.head_dim,
            config.key_bits,
            config.value_bits,
        );
        KVCacheState {
            inner: Arc::new(Mutex::new(inner)),
            config,
            batch_size,
            num_heads,
        }
    }

    fn append(&mut self, keys: PyReadonlyArray4<f32>, values: PyReadonlyArray4<f32>) -> PyResult<()> {
        let keys_view = keys.as_array();
        let values_view = values.as_array();
        
        let shape = keys_view.shape();
        if shape.len() != 4 || shape[0] != self.batch_size || shape[1] != self.num_heads || shape[3] != self.config.head_dim {
            return Err(TQError::ShapeMismatch {
                expected: vec![self.batch_size, self.num_heads, shape[2], self.config.head_dim],
                found: shape.to_vec(),
            }.into());
        }
        
        let seq_len = shape[2];
        let mut inner = self.inner.lock().unwrap();
        
        for b in 0..self.batch_size {
            for h in 0..self.num_heads {
                for s in 0..seq_len {
                    let k_slice = keys_view.slice(ndarray::s![b, h, s, ..]).to_slice().unwrap();
                    let v_slice = values_view.slice(ndarray::s![b, h, s, ..]).to_slice().unwrap();
                    inner.append(k_slice, v_slice, b, h, s)?;
                }
            }
        }
        
        Ok(())
    }
    
    #[getter]
    fn num_tokens(&self) -> usize {
        self.inner.lock().unwrap().get_num_tokens()
    }

    fn attention_scores<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray4<f32>,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let query_view = query.as_array();
        let shape = query_view.shape();
        
        if shape.len() != 4 || shape[0] != self.batch_size || shape[1] != self.num_heads || shape[2] != 1 || shape[3] != self.config.head_dim {
            return Err(TQError::ShapeMismatch {
                expected: vec![self.batch_size, self.num_heads, 1, self.config.head_dim],
                found: shape.to_vec(),
            }.into());
        }
        
        let inner = self.inner.lock().unwrap();
        let num_tokens = inner.get_num_tokens();
        
        // We need to transform the query too if keys are transformed
        let mut q_transformed = query_view.to_owned();
        for b in 0..self.batch_size {
            for h in 0..self.num_heads {
                let mut q_slice = q_transformed.slice_mut(ndarray::s![b, h, 0, ..]);
                let q_data = q_slice.as_slice_mut().unwrap();
                crate::core::rotation::fwht_inplace(q_data)?;
            }
        }

        let scores_vec = crate::core::attention::attention_scores_packed(
            q_transformed.as_slice().unwrap(),
            &inner,
        )?;
        
        let scores_array = ndarray::Array4::from_shape_vec(
            (self.batch_size, self.num_heads, 1, num_tokens),
            scores_vec
        ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(scores_array.to_pyarray_bound(py))
    }
}

#[pymodule]
fn _turboquant(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KVCodecConfig>()?;
    m.add_class::<KVCodec>()?;
    m.add_class::<KVCacheState>()?;
    Ok(())
}
