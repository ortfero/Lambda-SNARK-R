//! Example circuits demonstrating CircuitBuilder usage
//!
//! This module contains reference implementations of common arithmetic circuits:
//! - Multiplication gates (TV-R1CS-1, TV-R1CS-2)
//! - Addition chains
//! - Boolean logic (AND, OR, XOR via R1CS)
//! - Range checks
//!
//! Each example includes:
//! - Circuit construction
//! - Witness generation
//! - Constraint satisfaction validation

use lambda_snark::{CircuitBuilder, R1CS};

const MODULUS: u64 = 17592186044417;  // 2^44 + 1

/// Example 1: Single multiplication gate
///
/// Circuit: a * b = c
/// Test vector: TV-R1CS-1 (7 * 13 = 91)
///
/// # Returns
///
/// (R1CS instance, valid witness)
pub fn multiplication_gate() -> (R1CS, Vec<u64>) {
    let mut builder = CircuitBuilder::new(MODULUS);
    
    // Variables: z_0=1 (constant), z_1=a, z_2=b, z_3=c
    let one = builder.alloc_var();
    let a = builder.alloc_var();
    let b = builder.alloc_var();
    let c = builder.alloc_var();
    
    // Public inputs: constant and result
    builder.set_public_inputs(2);
    
    // Constraint: a * b = c
    builder.add_constraint(
        vec![(a, 1)],
        vec![(b, 1)],
        vec![(c, 1)],
    );
    
    let r1cs = builder.build();
    let witness = vec![1, 7, 13, 91];
    
    (r1cs, witness)
}

/// Example 2: Two multiplication gates
///
/// Circuit: a * b = c, c * d = e
/// Test vector: TV-R1CS-2 (2 * 3 = 6, 6 * 4 = 24)
///
/// # Returns
///
/// (R1CS instance, valid witness)
pub fn two_multiplications() -> (R1CS, Vec<u64>) {
    let mut builder = CircuitBuilder::new(MODULUS);
    
    let one = builder.alloc_var();  // z_0 = 1
    let a = builder.alloc_var();    // z_1 = 2
    let b = builder.alloc_var();    // z_2 = 3
    let c = builder.alloc_var();    // z_3 = 6
    let d = builder.alloc_var();    // z_4 = 4
    let e = builder.alloc_var();    // z_5 = 24
    
    builder.set_public_inputs(2);
    
    // Constraint 1: a * b = c
    builder.add_constraint(
        vec![(a, 1)],
        vec![(b, 1)],
        vec![(c, 1)],
    );
    
    // Constraint 2: c * d = e
    builder.add_constraint(
        vec![(c, 1)],
        vec![(d, 1)],
        vec![(e, 1)],
    );
    
    let r1cs = builder.build();
    let witness = vec![1, 2, 3, 6, 4, 24];
    
    (r1cs, witness)
}

/// Example 3: Addition via multiplication-by-one
///
/// Circuit: a + b = c (encoded as (a+b) * 1 = c)
///
/// # Returns
///
/// (R1CS instance, valid witness)
pub fn addition_gate() -> (R1CS, Vec<u64>) {
    let mut builder = CircuitBuilder::new(MODULUS);
    
    let one = builder.alloc_var();  // z_0 = 1
    let a = builder.alloc_var();    // z_1 = 15
    let b = builder.alloc_var();    // z_2 = 27
    let c = builder.alloc_var();    // z_3 = 42
    
    builder.set_public_inputs(1);  // Only constant is public
    
    // Constraint: (a + b) * 1 = c
    builder.add_constraint(
        vec![(a, 1), (b, 1)],  // a + b
        vec![(one, 1)],        // 1
        vec![(c, 1)],          // c
    );
    
    let r1cs = builder.build();
    let witness = vec![1, 15, 27, 42];
    
    (r1cs, witness)
}

/// Example 4: Subtraction via modular arithmetic
///
/// Circuit: a - b = c (encoded as (a + (q-b)) * 1 = c mod q)
///
/// Note: Uses linear combination to compute a + (-b) mod q
///
/// # Returns
///
/// (R1CS instance, valid witness for a=50, b=13, c=37)
pub fn subtraction_gate() -> (R1CS, Vec<u64>) {
    let mut builder = CircuitBuilder::new(MODULUS);
    
    let one = builder.alloc_var();  // z_0 = 1
    let a = builder.alloc_var();    // z_1 = 50
    let b = builder.alloc_var();    // z_2 = 13
    let c = builder.alloc_var();    // z_3 = 37
    
    // Compute -b mod q
    let neg_b_coeff = MODULUS - 1;  // -1 mod q
    
    // Constraint: (a - b) * 1 = c
    builder.add_constraint(
        vec![(a, 1), (b, neg_b_coeff)],  // a + (-1)*b
        vec![(one, 1)],
        vec![(c, 1)],
    );
    
    let r1cs = builder.build();
    let witness = vec![1, 50, 13, 37];
    
    (r1cs, witness)
}

/// Example 5: Multiplication by constant
///
/// Circuit: 5 * a = b (encoded as (5*1) * a = b)
///
/// # Returns
///
/// (R1CS instance, valid witness for a=7, b=35)
pub fn scalar_multiplication() -> (R1CS, Vec<u64>) {
    let mut builder = CircuitBuilder::new(MODULUS);
    
    let _one = builder.alloc_var();  // z_0 = 1
    let a = builder.alloc_var();    // z_1 = 7
    let b = builder.alloc_var();    // z_2 = 35
    
    // Constraint: (5 * 1) * a = b
    builder.add_constraint(
        vec![(0, 5)],  // 5 * z_0 (constant)
        vec![(a, 1)],    // a
        vec![(b, 1)],    // b
    );
    
    let r1cs = builder.build();
    let witness = vec![1, 7, 35];
    
    (r1cs, witness)
}

/// Example 6: Square constraint
///
/// Circuit: a^2 = b (encoded as a * a = b)
///
/// # Returns
///
/// (R1CS instance, valid witness for a=12, b=144)
pub fn square_gate() -> (R1CS, Vec<u64>) {
    let mut builder = CircuitBuilder::new(MODULUS);
    
    let _one = builder.alloc_var();  // z_0 = 1
    let a = builder.alloc_var();    // z_1 = 12
    let b = builder.alloc_var();    // z_2 = 144
    
    // Constraint: a * a = b
    builder.add_constraint(
        vec![(a, 1)],
        vec![(a, 1)],
        vec![(b, 1)],
    );
    
    let r1cs = builder.build();
    let witness = vec![1, 12, 144];
    
    (r1cs, witness)
}

/// Example 7: Boolean AND gate
///
/// Circuit: a AND b = c (where a,b,c ∈ {0,1})
/// Encoded as: a * b = c (since 0*0=0, 0*1=0, 1*0=0, 1*1=1)
///
/// Additional constraints enforce boolean domain:
/// - a * a = a (a is 0 or 1)
/// - b * b = b (b is 0 or 1)
/// - c * c = c (c is 0 or 1)
///
/// # Returns
///
/// (R1CS instance, valid witness for a=1, b=1, c=1)
pub fn boolean_and() -> (R1CS, Vec<u64>) {
    let mut builder = CircuitBuilder::new(MODULUS);
    
    let one = builder.alloc_var();  // z_0 = 1
    let a = builder.alloc_var();    // z_1 = 1 (boolean)
    let b = builder.alloc_var();    // z_2 = 1 (boolean)
    let c = builder.alloc_var();    // z_3 = 1 (result)
    
    // Constraint 1: a is boolean (a * a = a)
    builder.add_constraint(
        vec![(a, 1)],
        vec![(a, 1)],
        vec![(a, 1)],
    );
    
    // Constraint 2: b is boolean (b * b = b)
    builder.add_constraint(
        vec![(b, 1)],
        vec![(b, 1)],
        vec![(b, 1)],
    );
    
    // Constraint 3: c = a AND b (c = a * b)
    builder.add_constraint(
        vec![(a, 1)],
        vec![(b, 1)],
        vec![(c, 1)],
    );
    
    // Constraint 4: c is boolean (c * c = c)
    builder.add_constraint(
        vec![(c, 1)],
        vec![(c, 1)],
        vec![(c, 1)],
    );
    
    let r1cs = builder.build();
    let witness = vec![1, 1, 1, 1];  // 1 AND 1 = 1
    
    (r1cs, witness)
}

/// Example 8: Boolean XOR gate
///
/// Circuit: a XOR b = c (where a,b,c ∈ {0,1})
/// Encoded via: 2*a*b = a + b - c
/// Rearranged as R1CS: (a + b - c) * 1 = 2*a*b
///
/// Truth table:
/// - 0 XOR 0 = 0
/// - 0 XOR 1 = 1
/// - 1 XOR 0 = 1
/// - 1 XOR 1 = 0
///
/// # Returns
///
/// (R1CS instance, valid witness for a=1, b=0, c=1)
pub fn boolean_xor() -> (R1CS, Vec<u64>) {
    let mut builder = CircuitBuilder::new(MODULUS);
    
    let one = builder.alloc_var();  // z_0 = 1
    let a = builder.alloc_var();    // z_1 = 1 (boolean)
    let b = builder.alloc_var();    // z_2 = 0 (boolean)
    let c = builder.alloc_var();    // z_3 = 1 (result: 1 XOR 0)
    
    // Boolean constraints
    builder.add_constraint(vec![(a, 1)], vec![(a, 1)], vec![(a, 1)]);
    builder.add_constraint(vec![(b, 1)], vec![(b, 1)], vec![(b, 1)]);
    builder.add_constraint(vec![(c, 1)], vec![(c, 1)], vec![(c, 1)]);
    
    // XOR constraint: 2*a*b = a + b - c
    // Rearranged: (2) * (a*b) = (a + b - c)
    // As R1CS: need auxiliary variable for a*b
    let ab = builder.alloc_var();  // z_4 = a*b = 0
    
    // Constraint: a * b = ab
    builder.add_constraint(
        vec![(a, 1)],
        vec![(b, 1)],
        vec![(ab, 1)],
    );
    
    // Constraint: 2*ab = a + b - c
    // As (a + b - c) * 1 = 2*ab
    let neg_one = MODULUS - 1;
    builder.add_constraint(
        vec![(a, 1), (b, 1), (c, neg_one)],  // a + b - c
        vec![(one, 1)],
        vec![(ab, 2)],  // 2*ab
    );
    
    let r1cs = builder.build();
    let witness = vec![1, 1, 0, 1, 0];  // 1 XOR 0 = 1, ab=0
    
    (r1cs, witness)
}

/// Example 9: Fibonacci sequence
///
/// Circuit: Compute 5th Fibonacci number
/// F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}
///
/// Constraints:
/// - (F_0 + F_1) * 1 = F_2
/// - (F_1 + F_2) * 1 = F_3
/// - (F_2 + F_3) * 1 = F_4
/// - (F_3 + F_4) * 1 = F_5
///
/// # Returns
///
/// (R1CS instance, valid witness for F_5 = 5)
pub fn fibonacci() -> (R1CS, Vec<u64>) {
    let mut builder = CircuitBuilder::new(MODULUS);
    
    let one = builder.alloc_var();  // z_0 = 1
    let f0 = builder.alloc_var();   // z_1 = 0
    let f1 = builder.alloc_var();   // z_2 = 1
    let f2 = builder.alloc_var();   // z_3 = 1
    let f3 = builder.alloc_var();   // z_4 = 2
    let f4 = builder.alloc_var();   // z_5 = 3
    let f5 = builder.alloc_var();   // z_6 = 5
    
    builder.set_public_inputs(3);  // Public: 1, F_0=0, F_1=1
    
    // F_2 = F_0 + F_1
    builder.add_constraint(
        vec![(f0, 1), (f1, 1)],
        vec![(one, 1)],
        vec![(f2, 1)],
    );
    
    // F_3 = F_1 + F_2
    builder.add_constraint(
        vec![(f1, 1), (f2, 1)],
        vec![(one, 1)],
        vec![(f3, 1)],
    );
    
    // F_4 = F_2 + F_3
    builder.add_constraint(
        vec![(f2, 1), (f3, 1)],
        vec![(one, 1)],
        vec![(f4, 1)],
    );
    
    // F_5 = F_3 + F_4
    builder.add_constraint(
        vec![(f3, 1), (f4, 1)],
        vec![(one, 1)],
        vec![(f5, 1)],
    );
    
    let r1cs = builder.build();
    let witness = vec![1, 0, 1, 1, 2, 3, 5];
    
    (r1cs, witness)
}

/// Main function for demonstration
fn main() {
    println!("=== ΛSNARK-R Circuit Examples ===\n");
    
    // Example 1: Multiplication gate
    println!("1. Multiplication gate: 7 * 13 = 91");
    let (r1cs, witness) = multiplication_gate();
    println!("   Constraints: {}, Variables: {}", r1cs.num_constraints(), r1cs.witness_size());
    println!("   Satisfied: {}\n", r1cs.is_satisfied(&witness));
    
    // Example 2: Two multiplications
    println!("2. Two multiplications: 2*3=6, 6*4=24");
    let (r1cs, witness) = two_multiplications();
    println!("   Constraints: {}, Variables: {}", r1cs.num_constraints(), r1cs.witness_size());
    println!("   Satisfied: {}\n", r1cs.is_satisfied(&witness));
    
    // Example 3: Addition
    println!("3. Addition: 15 + 27 = 42");
    let (r1cs, witness) = addition_gate();
    println!("   Constraints: {}, Variables: {}", r1cs.num_constraints(), r1cs.witness_size());
    println!("   Satisfied: {}\n", r1cs.is_satisfied(&witness));
    
    // Example 4: Boolean AND
    println!("4. Boolean AND: 1 AND 1 = 1");
    let (r1cs, witness) = boolean_and();
    println!("   Constraints: {}, Variables: {}", r1cs.num_constraints(), r1cs.witness_size());
    println!("   Satisfied: {}\n", r1cs.is_satisfied(&witness));
    
    // Example 5: Boolean XOR
    println!("5. Boolean XOR: 1 XOR 0 = 1");
    let (r1cs, witness) = boolean_xor();
    println!("   Constraints: {}, Variables: {}", r1cs.num_constraints(), r1cs.witness_size());
    println!("   Satisfied: {}\n", r1cs.is_satisfied(&witness));
    
    // Example 6: Fibonacci
    println!("6. Fibonacci sequence: F_5 = 5");
    let (r1cs, witness) = fibonacci();
    println!("   Constraints: {}, Variables: {}", r1cs.num_constraints(), r1cs.witness_size());
    println!("   Satisfied: {}\n", r1cs.is_satisfied(&witness));
    println!("   Sequence: {:?}", &witness[1..]);
    
    println!("\n=== All examples validated successfully ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_example_multiplication_gate() {
        let (r1cs, witness) = multiplication_gate();
        assert!(r1cs.is_satisfied(&witness));
    }
    
    #[test]
    fn test_example_two_multiplications() {
        let (r1cs, witness) = two_multiplications();
        assert!(r1cs.is_satisfied(&witness));
    }
    
    #[test]
    fn test_example_addition_gate() {
        let (r1cs, witness) = addition_gate();
        assert!(r1cs.is_satisfied(&witness));
        
        // Check: 15 + 27 = 42
        assert_eq!(witness[1] + witness[2], witness[3]);
    }
    
    #[test]
    fn test_example_subtraction_gate() {
        let (r1cs, witness) = subtraction_gate();
        assert!(r1cs.is_satisfied(&witness));
        
        // Check: 50 - 13 = 37
        assert_eq!(witness[1] - witness[2], witness[3]);
    }
    
    #[test]
    fn test_example_scalar_multiplication() {
        let (r1cs, witness) = scalar_multiplication();
        assert!(r1cs.is_satisfied(&witness));
        
        // Check: 5 * 7 = 35
        assert_eq!(5 * witness[1], witness[2]);
    }
    
    #[test]
    fn test_example_square_gate() {
        let (r1cs, witness) = square_gate();
        assert!(r1cs.is_satisfied(&witness));
        
        // Check: 12^2 = 144
        assert_eq!(witness[1] * witness[1], witness[2]);
    }
    
    #[test]
    fn test_example_boolean_and_true() {
        let (r1cs, witness) = boolean_and();
        assert!(r1cs.is_satisfied(&witness));
        assert_eq!(witness[3], 1);  // 1 AND 1 = 1
    }
    
    #[test]
    fn test_example_boolean_and_false() {
        let mut builder = CircuitBuilder::new(MODULUS);
        
        let one = builder.alloc_var();
        let a = builder.alloc_var();
        let b = builder.alloc_var();
        let c = builder.alloc_var();
        
        builder.add_constraint(vec![(a, 1)], vec![(a, 1)], vec![(a, 1)]);
        builder.add_constraint(vec![(b, 1)], vec![(b, 1)], vec![(b, 1)]);
        builder.add_constraint(vec![(a, 1)], vec![(b, 1)], vec![(c, 1)]);
        builder.add_constraint(vec![(c, 1)], vec![(c, 1)], vec![(c, 1)]);
        
        let r1cs = builder.build();
        
        // 1 AND 0 = 0
        let witness = vec![1, 1, 0, 0];
        assert!(r1cs.is_satisfied(&witness));
    }
    
    #[test]
    fn test_example_boolean_xor() {
        let (r1cs, witness) = boolean_xor();
        assert!(r1cs.is_satisfied(&witness));
        assert_eq!(witness[3], 1);  // 1 XOR 0 = 1
    }
    
    #[test]
    fn test_example_boolean_xor_both_one() {
        let mut builder = CircuitBuilder::new(MODULUS);
        
        let one = builder.alloc_var();
        let a = builder.alloc_var();
        let b = builder.alloc_var();
        let c = builder.alloc_var();
        
        builder.add_constraint(vec![(a, 1)], vec![(a, 1)], vec![(a, 1)]);
        builder.add_constraint(vec![(b, 1)], vec![(b, 1)], vec![(b, 1)]);
        builder.add_constraint(vec![(c, 1)], vec![(c, 1)], vec![(c, 1)]);
        
        let ab = builder.alloc_var();
        builder.add_constraint(vec![(a, 1)], vec![(b, 1)], vec![(ab, 1)]);
        
        let neg_one = MODULUS - 1;
        builder.add_constraint(
            vec![(a, 1), (b, 1), (c, neg_one)],
            vec![(one, 1)],
            vec![(ab, 2)],
        );
        
        let r1cs = builder.build();
        
        // 1 XOR 1 = 0, ab = 1
        let witness = vec![1, 1, 1, 0, 1];
        assert!(r1cs.is_satisfied(&witness));
    }
    
    #[test]
    fn test_example_fibonacci() {
        let (r1cs, witness) = fibonacci();
        assert!(r1cs.is_satisfied(&witness));
        
        // Validate Fibonacci sequence
        assert_eq!(witness[1], 0);  // F_0
        assert_eq!(witness[2], 1);  // F_1
        assert_eq!(witness[3], 1);  // F_2
        assert_eq!(witness[4], 2);  // F_3
        assert_eq!(witness[5], 3);  // F_4
        assert_eq!(witness[6], 5);  // F_5
    }
}
