//! Sparse matrix implementation in CSR (Compressed Sparse Row) format.
//!
//! CSR format is memory-efficient for sparse matrices (typical R1CS circuits are <1% dense).
//! Memory: O(nnz) instead of O(m·n) for dense representation.
//!
//! # Format
//!
//! A sparse matrix M ∈ F_q^{m×n} is stored as:
//! - `row_ptr`: Offset array, row_ptr\[i\] = start index in col_indices for row i
//! - `col_indices`: Column indices of non-zero entries
//! - `values`: Non-zero values (field elements)
//!
//! # Example
//!
//! ```text
//! Dense matrix:
//! [0, 1, 0, 0]
//! [0, 0, 1, 0]
//! [0, 0, 0, 1]
//!
//! CSR representation:
//! row_ptr = [0, 1, 2, 3]
//! col_indices = [1, 2, 3]
//! values = [1, 1, 1]
//! ```

use std::collections::HashMap;

use crate::arith::{add_mod, mul_mod};

/// Sparse matrix in CSR (Compressed Sparse Row) format.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix {
    /// Number of rows (m for R1CS constraints)
    rows: usize,

    /// Number of columns (n for R1CS witness size)
    cols: usize,

    /// Row pointer array: row_ptr[i] = start index in col_indices for row i
    /// Length: rows + 1 (last element = nnz)
    row_ptr: Vec<usize>,

    /// Column indices of non-zero entries
    /// Length: nnz (number of non-zeros)
    col_indices: Vec<usize>,

    /// Non-zero values (field elements mod q)
    /// Length: nnz
    values: Vec<u64>,
}

impl SparseMatrix {
    /// Create new sparse matrix from CSR components.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `row_ptr` - Row pointer array (length = rows + 1)
    /// * `col_indices` - Column indices (length = nnz)
    /// * `values` - Non-zero values (length = nnz)
    ///
    /// # Panics
    ///
    /// Panics if CSR invariants are violated:
    /// - row_ptr.len() != rows + 1
    /// - col_indices.len() != values.len()
    /// - row_ptr[i] > row_ptr[i+1] (non-monotonic)
    /// - col_indices contains index >= cols
    pub fn new(
        rows: usize,
        cols: usize,
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<u64>,
    ) -> Self {
        // Validate CSR invariants
        assert_eq!(row_ptr.len(), rows + 1, "row_ptr must have length rows + 1");
        assert_eq!(
            col_indices.len(),
            values.len(),
            "col_indices and values must have same length"
        );

        // Check monotonicity
        for i in 0..rows {
            assert!(
                row_ptr[i] <= row_ptr[i + 1],
                "row_ptr must be monotonically increasing"
            );
        }

        // Check column indices are in bounds
        for &col in &col_indices {
            assert!(
                col < cols,
                "Column index {} out of bounds (cols={})",
                col,
                cols
            );
        }

        SparseMatrix {
            rows,
            cols,
            row_ptr,
            col_indices,
            values,
        }
    }

    /// Create sparse matrix from dense row representation.
    ///
    /// # Example
    ///
    /// ```
    /// use lambda_snark::sparse_matrix::SparseMatrix;
    ///
    /// let rows = vec![
    ///     vec![0, 1, 0, 0],
    ///     vec![0, 0, 1, 0],
    /// ];
    /// let matrix = SparseMatrix::from_dense(&rows);
    /// assert_eq!(matrix.rows(), 2);
    /// assert_eq!(matrix.cols(), 4);
    /// assert_eq!(matrix.nnz(), 2);
    /// ```
    pub fn from_dense(rows: &[Vec<u64>]) -> Self {
        if rows.is_empty() {
            return Self::new(0, 0, vec![0], vec![], vec![]);
        }

        let m = rows.len();
        let n = rows[0].len();

        let mut row_ptr = Vec::with_capacity(m + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptr.push(0);

        for row in rows {
            assert_eq!(row.len(), n, "All rows must have same length");

            for (col, &val) in row.iter().enumerate() {
                if val != 0 {
                    col_indices.push(col);
                    values.push(val);
                }
            }

            row_ptr.push(col_indices.len());
        }

        Self::new(m, n, row_ptr, col_indices, values)
    }

    /// Create sparse matrix from HashMap of (row, col) -> value.
    ///
    /// # Example
    ///
    /// ```
    /// use lambda_snark::sparse_matrix::SparseMatrix;
    /// use std::collections::HashMap;
    ///
    /// let mut entries = HashMap::new();
    /// entries.insert((0, 1), 5);
    /// entries.insert((1, 2), 7);
    ///
    /// let matrix = SparseMatrix::from_map(2, 4, &entries);
    /// assert_eq!(matrix.get(0, 1), 5);
    /// assert_eq!(matrix.get(1, 2), 7);
    /// ```
    pub fn from_map(rows: usize, cols: usize, entries: &HashMap<(usize, usize), u64>) -> Self {
        let mut row_data: Vec<Vec<(usize, u64)>> = vec![Vec::new(); rows];

        for (&(r, c), &val) in entries {
            assert!(r < rows, "Row {} out of bounds", r);
            assert!(c < cols, "Column {} out of bounds", c);
            if val != 0 {
                row_data[r].push((c, val));
            }
        }

        // Sort each row by column index
        for row in &mut row_data {
            row.sort_by_key(|&(c, _)| c);
        }

        let mut row_ptr = Vec::with_capacity(rows + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptr.push(0);

        for row in row_data {
            for (c, val) in row {
                col_indices.push(c);
                values.push(val);
            }
            row_ptr.push(col_indices.len());
        }

        Self::new(rows, cols, row_ptr, col_indices, values)
    }

    /// Get element at (row, col). Returns 0 if not stored (implicit zero).
    ///
    /// Time complexity: O(nnz_row) where nnz_row is number of non-zeros in row.
    pub fn get(&self, row: usize, col: usize) -> u64 {
        assert!(row < self.rows, "Row {} out of bounds", row);
        assert!(col < self.cols, "Column {} out of bounds", col);

        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        for i in start..end {
            if self.col_indices[i] == col {
                return self.values[i];
            } else if self.col_indices[i] > col {
                // Columns are sorted, so we can early exit
                return 0;
            }
        }

        0
    }

    /// Matrix-vector product: result[i] = Σ_j M[i,j] * v[j] mod q
    ///
    /// # Arguments
    ///
    /// * `v` - Input vector (length = cols)
    /// * `modulus` - Field modulus q
    ///
    /// # Returns
    ///
    /// Result vector (length = rows)
    ///
    /// # Panics
    ///
    /// Panics if v.len() != cols
    ///
    /// # Example
    ///
    /// ```
    /// use lambda_snark::sparse_matrix::SparseMatrix;
    ///
    /// let rows = vec![
    ///     vec![0, 1, 0, 0],
    ///     vec![0, 0, 1, 0],
    /// ];
    /// let matrix = SparseMatrix::from_dense(&rows);
    /// let v = vec![1, 7, 13, 91];
    /// let result = matrix.mul_vec(&v, 1000);
    /// assert_eq!(result, vec![7, 13]);
    /// ```
    pub fn mul_vec(&self, v: &[u64], modulus: u64) -> Vec<u64> {
        assert_eq!(
            v.len(),
            self.cols,
            "Vector length {} must equal matrix cols {}",
            v.len(),
            self.cols
        );

        let mut result = vec![0u64; self.rows];

        for row in 0..self.rows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            let mut sum = 0u64;

            for i in start..end {
                let col = self.col_indices[i];
                let val = self.values[i];

                // Constant-time modular multiply and accumulate
                let term = mul_mod(val % modulus, v[col] % modulus, modulus);
                sum = add_mod(sum, term, modulus);
            }

            result[row] = sum;
        }

        result
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get row pointer array (for advanced use).
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Get column indices (for advanced use).
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Get values (for advanced use).
    pub fn values(&self) -> &[u64] {
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_matrix() {
        let matrix = SparseMatrix::new(0, 0, vec![0], vec![], vec![]);
        assert_eq!(matrix.rows(), 0);
        assert_eq!(matrix.cols(), 0);
        assert_eq!(matrix.nnz(), 0);
    }

    #[test]
    fn test_from_dense_simple() {
        let rows = vec![vec![0, 1, 0, 0], vec![0, 0, 1, 0], vec![0, 0, 0, 1]];

        let matrix = SparseMatrix::from_dense(&rows);

        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 4);
        assert_eq!(matrix.nnz(), 3);

        assert_eq!(matrix.get(0, 1), 1);
        assert_eq!(matrix.get(1, 2), 1);
        assert_eq!(matrix.get(2, 3), 1);

        assert_eq!(matrix.get(0, 0), 0);
        assert_eq!(matrix.get(1, 1), 0);
    }

    #[test]
    fn test_from_map() {
        let mut entries = HashMap::new();
        entries.insert((0, 1), 5);
        entries.insert((1, 2), 7);
        entries.insert((1, 0), 3);

        let matrix = SparseMatrix::from_map(2, 4, &entries);

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 4);
        assert_eq!(matrix.nnz(), 3);

        assert_eq!(matrix.get(0, 1), 5);
        assert_eq!(matrix.get(1, 0), 3);
        assert_eq!(matrix.get(1, 2), 7);
    }

    #[test]
    fn test_mul_vec_simple() {
        let rows = vec![vec![0, 1, 0, 0], vec![0, 0, 1, 0], vec![0, 0, 0, 1]];

        let matrix = SparseMatrix::from_dense(&rows);
        let v = vec![1, 7, 13, 91];
        let modulus = 1000;

        let result = matrix.mul_vec(&v, modulus);

        assert_eq!(result, vec![7, 13, 91]);
    }

    #[test]
    fn test_mul_vec_with_modulus() {
        let rows = vec![vec![2, 0], vec![0, 3]];

        let matrix = SparseMatrix::from_dense(&rows);
        let v = vec![100, 200];
        let modulus = 50;

        // 2*100 = 200 mod 50 = 0
        // 3*200 = 600 mod 50 = 0
        let result = matrix.mul_vec(&v, modulus);

        assert_eq!(result, vec![0, 0]);
    }

    #[test]
    fn test_mul_vec_r1cs_example() {
        // TV-R1CS-1: multiplication gate
        // A = [0, 1, 0, 0] (select a)
        let a_matrix = SparseMatrix::from_dense(&vec![vec![0, 1, 0, 0]]);

        let witness = vec![1, 7, 13, 91];
        let modulus = 17592186044417; // 2^44 + 1

        let result = a_matrix.mul_vec(&witness, modulus);
        assert_eq!(result, vec![7]); // Should select a=7
    }

    #[test]
    fn test_sparse_efficiency() {
        // 1000x1000 matrix with only 10 non-zeros
        let mut entries = HashMap::new();
        for i in 0..10 {
            entries.insert((i, i * 100), 42);
        }

        let matrix = SparseMatrix::from_map(1000, 1000, &entries);

        assert_eq!(matrix.rows(), 1000);
        assert_eq!(matrix.cols(), 1000);
        assert_eq!(matrix.nnz(), 10); // Only 10 stored!

        // Dense would need 1000*1000 = 1M entries
        // Sparse uses only 10 entries + overhead
        // Memory savings: ~99.999%
    }

    #[test]
    fn test_get_performance() {
        // Test that get() correctly handles sorted columns (early exit)
        let rows = vec![vec![0, 0, 0, 5, 0, 0, 0, 7, 0, 0]];

        let matrix = SparseMatrix::from_dense(&rows);

        // These should return quickly (before non-zero)
        assert_eq!(matrix.get(0, 0), 0);
        assert_eq!(matrix.get(0, 1), 0);

        // Non-zero retrieval
        assert_eq!(matrix.get(0, 3), 5);
        assert_eq!(matrix.get(0, 7), 7);

        // After last non-zero
        assert_eq!(matrix.get(0, 9), 0);
    }

    #[test]
    #[should_panic(expected = "row_ptr must have length rows + 1")]
    fn test_invalid_row_ptr_length() {
        SparseMatrix::new(2, 2, vec![0, 1], vec![0], vec![1]);
    }

    #[test]
    #[should_panic(expected = "col_indices and values must have same length")]
    fn test_mismatched_col_indices_values() {
        SparseMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1]);
    }

    #[test]
    #[should_panic(expected = "Column index")]
    fn test_col_index_out_of_bounds() {
        SparseMatrix::new(2, 2, vec![0, 1, 1], vec![5], vec![1]);
    }

    #[test]
    #[should_panic(expected = "row_ptr must be monotonically increasing")]
    fn test_non_monotonic_row_ptr() {
        SparseMatrix::new(2, 2, vec![0, 2, 1], vec![0, 1], vec![1, 2]);
    }

    #[test]
    fn test_zero_matrix() {
        let rows = vec![vec![0, 0], vec![0, 0]];

        let matrix = SparseMatrix::from_dense(&rows);

        assert_eq!(matrix.nnz(), 0);
        assert_eq!(matrix.get(0, 0), 0);
        assert_eq!(matrix.get(1, 1), 0);

        let v = vec![1, 2];
        let result = matrix.mul_vec(&v, 100);
        assert_eq!(result, vec![0, 0]);
    }
}
