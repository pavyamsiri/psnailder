//! Runtime parameter layouts for fitting models with fixed and free values.

use alloc::vec::Vec;
use core::{error::Error, fmt, result::Result};

/// A bound for one model parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParameterBound {
    /// A parameter that may vary over a closed interval.
    Interval { lower: f64, upper: f64 },
    /// A parameter that is held at one value.
    Fixed(f64),
}

/// Errors returned while constructing or using a [`ParameterLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutError {
    /// A bound contains a non-finite value.
    NonFiniteBound { index: usize },
    /// An interval has its lower endpoint above its upper endpoint.
    ReversedInterval { index: usize },
    /// A vector has a different length from the layout it is used with.
    WrongLength { expected: usize, actual: usize },
    /// A value supplied for a fixed parameter does not match the fixed value.
    FixedValueMismatch { index: usize },
    /// A free value lies outside its interval.
    OutsideBounds { index: usize },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteBound { index } => write!(f, "non-finite bound at index {index}"),
            Self::ReversedInterval { index } => write!(f, "reversed interval at index {index}"),
            Self::WrongLength { expected, actual } => {
                write!(f, "expected vector of length {expected}, got {actual}")
            }
            Self::FixedValueMismatch { index } => {
                write!(f, "fixed value mismatch at index {index}")
            }
            Self::OutsideBounds { index } => write!(f, "value outside bounds at index {index}"),
        }
    }
}

impl Error for LayoutError {}

/// The mapping between a compact free-parameter vector and a full model vector.
///
/// The order of free values is always the original parameter order. Fixed values
/// are stored in the layout and are restored by [`Self::unpack`].
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterLayout {
    template: Vec<f64>,
    free_indices: Vec<usize>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl ParameterLayout {
    /// Construct a layout from full-model bounds.
    ///
    /// # Errors
    /// Returns an error for non-finite values or reversed intervals.
    pub fn from_bounds(bounds: &[ParameterBound]) -> Result<Self, LayoutError> {
        let mut template = Vec::with_capacity(bounds.len());
        let mut free_indices = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();

        for (index, bound) in bounds.iter().copied().enumerate() {
            match bound {
                ParameterBound::Fixed(value) => {
                    if !value.is_finite() {
                        return Err(LayoutError::NonFiniteBound { index });
                    }
                    template.push(value);
                }
                ParameterBound::Interval {
                    lower: lb,
                    upper: ub,
                } => {
                    if !lb.is_finite() || !ub.is_finite() {
                        return Err(LayoutError::NonFiniteBound { index });
                    }
                    if lb > ub {
                        return Err(LayoutError::ReversedInterval { index });
                    }
                    template.push(lb);
                    free_indices.push(index);
                    lower.push(lb);
                    upper.push(ub);
                }
            }
        }

        Ok(Self {
            template,
            free_indices,
            lower,
            upper,
        })
    }

    /// Number of parameters in the full model vector.
    #[must_use]
    pub const fn full_len(&self) -> usize {
        self.template.len()
    }

    /// Number of parameters that remain free.
    #[must_use]
    pub const fn free_len(&self) -> usize {
        self.free_indices.len()
    }

    /// Lower bounds for the compact free vector.
    #[must_use]
    pub fn lower(&self) -> &Vec<f64> {
        &self.lower
    }

    /// Upper bounds for the compact free vector.
    #[must_use]
    pub fn upper(&self) -> &Vec<f64> {
        &self.upper
    }

    /// Convert a full vector into the compact free vector.
    ///
    /// # Errors
    /// Returns an error when the vector has the wrong length, contains a
    /// non-finite value, or disagrees with a fixed parameter.
    pub fn pack(&self, full: &[f64]) -> Result<Vec<f64>, LayoutError> {
        if full.len() != self.full_len() {
            return Err(LayoutError::WrongLength {
                expected: self.full_len(),
                actual: full.len(),
            });
        }
        for (index, (&value, &fixed)) in full.iter().zip(self.template.iter()).enumerate() {
            if !value.is_finite() {
                return Err(LayoutError::NonFiniteBound { index });
            }
            if !self.free_indices.contains(&index) && value.to_bits() != fixed.to_bits() {
                return Err(LayoutError::FixedValueMismatch { index });
            }
        }
        let free = self.free_indices.iter().map(|&index| full[index]).collect();
        Ok(free)
    }

    /// Expand a compact free vector into the full model vector.
    ///
    /// # Errors
    /// Returns an error when the vector has the wrong length or contains a
    /// non-finite or out-of-bounds free value.
    pub fn unpack(&self, free: &[f64]) -> Result<Vec<f64>, LayoutError> {
        if free.len() != self.free_len() {
            return Err(LayoutError::WrongLength {
                expected: self.free_len(),
                actual: free.len(),
            });
        }
        for (index, &value) in free.iter().enumerate() {
            if !value.is_finite() || value < self.lower[index] || value > self.upper[index] {
                return Err(LayoutError::OutsideBounds { index });
            }
        }
        let mut full = self.template.clone();
        for (value, &index) in free.iter().zip(self.free_indices.iter()) {
            full[index] = *value;
        }
        Ok(full)
    }
}

#[cfg(test)]
mod tests {
    use super::{LayoutError, ParameterBound, ParameterLayout};

    #[test]
    fn packs_and_unpacks_fixed_values() {
        let layout = ParameterLayout::from_bounds(&[
            ParameterBound::Interval {
                lower: 0.0,
                upper: 1.0,
            },
            ParameterBound::Fixed(4.0),
            ParameterBound::Interval {
                lower: -2.0,
                upper: 2.0,
            },
        ])
        .unwrap();
        assert_eq!(layout.full_len(), 3);
        assert_eq!(layout.free_len(), 2);
        assert_eq!(layout.pack(&[0.5, 4.0, -1.0]).unwrap(), [0.5, -1.0]);
        assert_eq!(layout.unpack(&[0.5, -1.0]).unwrap(), [0.5, 4.0, -1.0]);
    }

    #[test]
    fn validates_bounds_and_lengths() {
        assert_eq!(
            ParameterLayout::from_bounds(&[ParameterBound::Fixed(f64::NAN)]),
            Err(LayoutError::NonFiniteBound { index: 0 })
        );
        assert_eq!(
            ParameterLayout::from_bounds(&[ParameterBound::Interval {
                lower: 1.0,
                upper: 0.0
            }]),
            Err(LayoutError::ReversedInterval { index: 0 })
        );
    }
}
