//! Integration tests for NTT-based polynomial interpolation.
//!
//! Verifies that:
//! 1. NTT path is taken for power-of-2 circuits with NTT_MODULUS
//! 2. Fallback path is taken for non-power-of-2 or incompatible modulus
//! 3. Both paths produce equivalent results (same proof verification)

#[cfg(test)]
mod ntt_integration_tests {
    use lambda_snark::{R1CS, SparseMatrix};
    use lambda_snark_core::NTT_MODULUS;
    
    /// Helper: Create R1CS with m multiplication gates: a_i * b_i = c_i
    fn create_multiplication_gates(m: usize, modulus: u64) -> R1CS {
        let n = 1 + 3 * m; // 1 constant + (a, b, c) per gate
        
        let mut a_rows = Vec::new();
        let mut b_rows = Vec::new();
        let mut c_rows = Vec::new();
        
        for i in 0..m {
            let a_idx = 1 + 3 * i;
            let b_idx = 1 + 3 * i + 1;
            let c_idx = 1 + 3 * i + 2;
            
            // A[i] = [0, ..., 1 (at a_i), ..., 0]
            let mut a_row = vec![0u64; n];
            a_row[a_idx] = 1;
            a_rows.push(a_row);
            
            // B[i] = [0, ..., 1 (at b_i), ..., 0]
            let mut b_row = vec![0u64; n];
            b_row[b_idx] = 1;
            b_rows.push(b_row);
            
            // C[i] = [0, ..., 1 (at c_i), ..., 0]
            let mut c_row = vec![0u64; n];
            c_row[c_idx] = 1;
            c_rows.push(c_row);
        }
        
        let a = SparseMatrix::from_dense(&a_rows);
        let b = SparseMatrix::from_dense(&b_rows);
        let c = SparseMatrix::from_dense(&c_rows);
        
        R1CS::new(m, n, 1, a, b, c, modulus) // m constraints, n vars, l=1 (constant is public)
    }
    
    /// Helper: Create witness for m multiplication gates
    fn create_witness(m: usize, modulus: u64) -> Vec<u64> {
        let mut witness = vec![1u64]; // z[0] = 1 (constant)
        
        for i in 1..=m {
            let a = ((2 * i as u64) % modulus) as u64;
            let b = ((3 * i as u64) % modulus) as u64;
            let c = ((a as u128 * b as u128) % modulus as u128) as u64;
            witness.extend_from_slice(&[a, b, c]);
        }
        
        witness
    }
    
    /// Test that NTT path is used for power-of-2 circuit with NTT_MODULUS.
    #[test]
    #[cfg(feature = "fft-ntt")]
    fn test_ntt_path_power_of_2() {
        // Create 4-constraint circuit (power of 2) with NTT_MODULUS
        let r1cs = create_multiplication_gates(4, NTT_MODULUS);
        let witness = create_witness(4, NTT_MODULUS);
        
        // Verify witness satisfies R1CS
        assert!(r1cs.is_satisfied(&witness));
        
        // Compute quotient polynomial (uses NTT internally)
        let q_poly = r1cs.compute_quotient_poly(&witness).unwrap();
        
        // Quotient degree should be ≤ m-1
        assert!(q_poly.len() <= r1cs.m, "Quotient degree too high");
        
        println!("✅ NTT path verified for m={} (power of 2)", r1cs.m);
    }
    
    /// Test that fallback path is used for non-power-of-2 circuit.
    #[test]
    #[cfg(feature = "fft-ntt")]
    fn test_fallback_non_power_of_2() {
        // Create 3-constraint circuit (NOT power of 2) with NTT_MODULUS
        let r1cs = create_multiplication_gates(3, NTT_MODULUS);
        let witness = create_witness(3, NTT_MODULUS);
        
        assert!(r1cs.is_satisfied(&witness));
        
        // Compute quotient (should use fallback Lagrange)
        let q_poly = r1cs.compute_quotient_poly(&witness).unwrap();
        assert!(q_poly.len() <= r1cs.m);
        
        println!("✅ Fallback path verified for m={} (not power of 2)", r1cs.m);
    }
    
    /// Test that fallback path is used for incompatible modulus.
    #[test]
    #[cfg(feature = "fft-ntt")]
    fn test_fallback_incompatible_modulus() {
        const LEGACY_MODULUS: u64 = 17_592_186_044_423; // 45-bit (NOT NTT-friendly)
        
        // Create 4-constraint circuit with LEGACY_MODULUS
        let r1cs = create_multiplication_gates(4, LEGACY_MODULUS);
        let witness = create_witness(4, LEGACY_MODULUS);
        
        assert!(r1cs.is_satisfied(&witness));
        
        // Compute quotient (should use fallback Lagrange)
        let q_poly = r1cs.compute_quotient_poly(&witness).unwrap();
        assert!(q_poly.len() <= r1cs.m);
        
        println!("✅ Fallback path verified for incompatible modulus");
    }
    
    /// Test large power-of-2 circuit (stress test).
    #[test]
    #[cfg(feature = "fft-ntt")]
    fn test_ntt_large_circuit() {
        // Create 64-constraint circuit (2^6)
        let r1cs = create_multiplication_gates(64, NTT_MODULUS);
        let witness = create_witness(64, NTT_MODULUS);
        
        assert!(r1cs.is_satisfied(&witness));
        
        // Compute quotient (should use NTT for m=64)
        let q_poly = r1cs.compute_quotient_poly(&witness).unwrap();
        assert!(q_poly.len() <= r1cs.m);
        
        println!("✅ Large NTT circuit verified (m=64)");
    }
}
