//! R1CS integration tests.
//!
//! Tests R1CS constraint system with C++ backend through FFI.

use lambda_snark_core::r1cs::{SparseMatrix, R1CS};

const TEST_MODULUS: u64 = 17592186044417; // 2^44 + 1

#[test]
fn test_simple_multiplication() {
    // Constraint: a * b = c
    // Witness: z = [1, a, b, c] = [1, 7, 13, 91]
    //
    // R1CS encoding:
    // A = [[0, 1, 0, 0]]  (selects a)
    // B = [[0, 0, 1, 0]]  (selects b)
    // C = [[0, 0, 0, 1]]  (selects c)
    let a = SparseMatrix::from_entries(1, 4, vec![(0, 1, 1)]);
    let b = SparseMatrix::from_entries(1, 4, vec![(0, 2, 1)]);
    let c = SparseMatrix::from_entries(1, 4, vec![(0, 3, 1)]);

    let r1cs = R1CS::new(a, b, c, TEST_MODULUS).expect("Failed to create R1CS");

    assert_eq!(r1cs.num_constraints(), 1);
    assert_eq!(r1cs.num_variables(), 4);

    // Valid witness: 7 × 13 = 91
    assert!(r1cs.validate_witness(&[1, 7, 13, 91]).unwrap());

    // Invalid witness: 7 × 13 ≠ 92
    assert!(!r1cs.validate_witness(&[1, 7, 13, 92]).unwrap());
}

#[test]
fn test_two_constraints() {
    // Constraints:
    // 1. a * b = c
    // 2. c * 1 = c  (identity check)
    //
    // Witness: [1, a, b, c] = [1, 7, 13, 91]
    let a = SparseMatrix::from_entries(
        2,
        4,
        vec![
            (0, 1, 1), // A[0,1] = 1 (constraint 0: select a)
            (1, 3, 1), // A[1,3] = 1 (constraint 1: select c)
        ],
    );
    let b = SparseMatrix::from_entries(
        2,
        4,
        vec![
            (0, 2, 1), // B[0,2] = 1 (constraint 0: select b)
            (1, 0, 1), // B[1,0] = 1 (constraint 1: select constant 1)
        ],
    );
    let c = SparseMatrix::from_entries(
        2,
        4,
        vec![
            (0, 3, 1), // C[0,3] = 1 (constraint 0: select c)
            (1, 3, 1), // C[1,3] = 1 (constraint 1: select c)
        ],
    );

    let r1cs = R1CS::new(a, b, c, TEST_MODULUS).unwrap();

    assert_eq!(r1cs.num_constraints(), 2);
    assert_eq!(r1cs.num_variables(), 4);

    assert!(r1cs.validate_witness(&[1, 7, 13, 91]).unwrap());
}

#[test]
fn test_empty_constraints() {
    // Zero constraints — any witness should be valid
    let a = SparseMatrix::from_entries(0, 3, vec![]);
    let b = SparseMatrix::from_entries(0, 3, vec![]);
    let c = SparseMatrix::from_entries(0, 3, vec![]);

    let r1cs = R1CS::new(a, b, c, TEST_MODULUS).unwrap();

    assert_eq!(r1cs.num_constraints(), 0);
    assert_eq!(r1cs.num_variables(), 3);

    // Any witness is valid (no constraints to violate)
    assert!(r1cs.validate_witness(&[1, 42, 100]).unwrap());
    assert!(r1cs.validate_witness(&[1, 0, 0]).unwrap());
}

#[test]
fn test_witness_length_mismatch() {
    let a = SparseMatrix::from_entries(1, 4, vec![(0, 1, 1)]);
    let b = SparseMatrix::from_entries(1, 4, vec![(0, 2, 1)]);
    let c = SparseMatrix::from_entries(1, 4, vec![(0, 3, 1)]);

    let r1cs = R1CS::new(a, b, c, TEST_MODULUS).unwrap();

    // Witness too short
    assert!(r1cs.validate_witness(&[1, 2, 3]).is_err());

    // Witness too long
    assert!(r1cs.validate_witness(&[1, 2, 3, 4, 5]).is_err());
}

#[test]
fn test_dimension_mismatch() {
    // A has 1 row, B has 2 rows — should fail
    let a = SparseMatrix::from_entries(1, 3, vec![(0, 0, 1)]);
    let b = SparseMatrix::from_entries(2, 3, vec![(0, 0, 1)]);
    let c = SparseMatrix::from_entries(1, 3, vec![]);

    assert!(R1CS::new(a, b, c, TEST_MODULUS).is_err());
}

#[test]
fn test_modular_arithmetic() {
    // Test large values near modulus
    // (q-1) × (q-1) mod q = 1
    let a = SparseMatrix::from_entries(1, 4, vec![(0, 1, 1)]);
    let b = SparseMatrix::from_entries(1, 4, vec![(0, 2, 1)]);
    let c = SparseMatrix::from_entries(1, 4, vec![(0, 3, 1)]);

    let r1cs = R1CS::new(a, b, c, TEST_MODULUS).unwrap();

    let large_val = TEST_MODULUS - 1;
    // (q-1) × (q-1) = q^2 - 2q + 1 ≡ 1 (mod q)
    assert!(r1cs
        .validate_witness(&[1, large_val, large_val, 1])
        .unwrap());
}

#[test]
fn test_linear_combination() {
    // Constraint: (a + 2b) * c = d
    // Witness: [1, a, b, c, d] = [1, 3, 5, 7, 91]
    // (3 + 2×5) × 7 = 13 × 7 = 91
    let a = SparseMatrix::from_entries(
        1,
        5,
        vec![
            (0, 1, 1), // A[0,1] = 1 (select a)
            (0, 2, 2), // A[0,2] = 2 (select 2×b)
        ],
    );
    let b = SparseMatrix::from_entries(1, 5, vec![(0, 3, 1)]); // B[0,3] = 1 (select c)
    let c = SparseMatrix::from_entries(1, 5, vec![(0, 4, 1)]); // C[0,4] = 1 (select d)

    let r1cs = R1CS::new(a, b, c, TEST_MODULUS).unwrap();

    // Valid: (3 + 2×5) × 7 = 91
    assert!(r1cs.validate_witness(&[1, 3, 5, 7, 91]).unwrap());

    // Invalid: (3 + 2×5) × 7 ≠ 90
    assert!(!r1cs.validate_witness(&[1, 3, 5, 7, 90]).unwrap());
}

#[test]
fn test_tv1_multiplication() {
    // Test Vector 1: 7 × 13 = 91 (from TV-1)
    // This matches the multiplication constraint from test-vectors/tv-1-multiplication
    let a = SparseMatrix::from_entries(1, 4, vec![(0, 1, 1)]);
    let b = SparseMatrix::from_entries(1, 4, vec![(0, 2, 1)]);
    let c = SparseMatrix::from_entries(1, 4, vec![(0, 3, 1)]);

    let r1cs = R1CS::new(a, b, c, TEST_MODULUS).unwrap();

    // Witness from TV-1: [1, a=7, b=13, c=91]
    let witness = vec![1, 7, 13, 91];
    assert!(r1cs.validate_witness(&witness).unwrap());
}
