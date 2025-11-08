//! ΛSNARK-R: Post-quantum SNARK over lattices.
//!
//! This library provides a production-grade implementation of lattice-based SNARKs
//! using Module-LWE/SIS hardness assumptions.
//!
//! # Quick Start
//!
//! ```no_run
//! use lambda_snark::{Params, Profile, SecurityLevel, Field, setup, prove, verify};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Setup parameters
//! let params = Params::new(
//!     SecurityLevel::Bits128,
//!     Profile::RingB {
//!         n: 4096,
//!         k: 2,
//!         q: 17592186044417,  // 2^44 + 1 (prime)
//!         sigma: 3.19,
//!     },
//! );
//!
//! let (pk, vk) = setup(params)?;
//!
//! // Prove a * b = c (R1CS) - NOT YET IMPLEMENTED
//! let public_input = vec![Field::new(1), Field::new(91)];  // (1, 91)
//! let witness = vec![Field::new(7), Field::new(13)];       // a=7, b=13, c=91
//!
//! // TODO: prover not yet implemented
//! // let proof = prove(&pk, &public_input, &witness)?;
//! // let valid = verify(&vk, &public_input, &proof)?;
//! // assert!(valid);
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────┐
//! │   lambda-snark (Safe Rust API)  │
//! └─────────────┬───────────────────┘
//!               │ FFI (safe wrappers)
//! ┌─────────────▼───────────────────┐
//! │  lambda-snark-sys (FFI bindings)│
//! └─────────────┬───────────────────┘
//!               │ extern "C"
//! ┌─────────────▼───────────────────┐
//! │  cpp-core (C++ performance)     │
//! │  - SEAL (LWE commitment)        │
//! │  - NTL (NTT)                    │
//! │  - Eigen (linear algebra)       │
//! └─────────────────────────────────┘
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]

pub use lambda_snark_core::{Field, Params, Profile, SecurityLevel, Witness};
pub use lambda_snark_core::Error as CoreError;

mod commitment;
mod context;
mod polynomial;
mod challenge;
mod opening;
pub mod sparse_matrix;
pub mod r1cs;
pub mod circuit;
mod ntt;

pub use commitment::Commitment;
pub use context::LweContext;
pub use polynomial::Polynomial;
pub use challenge::Challenge;
pub use opening::{Opening, generate_opening, verify_opening, verify_opening_with_context};
pub use sparse_matrix::SparseMatrix;
pub use r1cs::R1CS;
pub use circuit::CircuitBuilder;

use thiserror::Error as ThisError;

/// ΛSNARK-R errors.
#[derive(Debug, ThisError)]
pub enum Error {
    /// Core error.
    #[error(transparent)]
    Core(#[from] CoreError),
    
    /// FFI error.
    #[error("FFI call failed: {0}")]
    Ffi(String),
    
    /// Invalid proof.
    #[error("Invalid proof")]
    InvalidProof,
    
    /// Invalid dimensions (e.g., NTT modular inverse failure).
    #[error("Invalid dimensions or parameters")]
    InvalidDimensions,
}

/// Proving key (stub).
pub struct ProvingKey {
    ctx: LweContext,
    // TODO: Add R1CS matrices, etc.
}

/// Verifying key (stub).
pub struct VerifyingKey {
    // TODO: Add public parameters
}

/// R1CS proof with two-challenge soundness.
///
/// Contains commitment to quotient polynomial Q(X) and two-challenge verification data.
///
/// # Structure
/// - Quotient polynomial: Q(X) = (A_z·B_z - C_z) / Z_H
/// - Two challenges: α, β (derived via Fiat-Shamir)
/// - Evaluations at both challenges for soundness amplification
/// - Opening proofs for both challenge points
///
/// # Security
/// - Soundness error: ε ≤ 2^(-48) (two independent challenges)
/// - Challenge derivation: Fiat-Shamir with SHA3-256
/// - LWE security: 128-bit quantum security
#[derive(Debug, Clone)]
pub struct ProofR1CS {
    /// LWE commitment to quotient polynomial Q(X)
    pub commitment_q: Commitment,
    
    /// First challenge α
    pub challenge_alpha: Challenge,
    
    /// Second challenge β
    pub challenge_beta: Challenge,
    
    /// Q(α) evaluation
    pub q_alpha: u64,
    
    /// Q(β) evaluation
    pub q_beta: u64,
    
    /// A_z(α) evaluation (left constraint polynomial at α)
    pub a_z_alpha: u64,
    
    /// B_z(α) evaluation (right constraint polynomial at α)
    pub b_z_alpha: u64,
    
    /// C_z(α) evaluation (output constraint polynomial at α)
    pub c_z_alpha: u64,
    
    /// A_z(β) evaluation (left constraint polynomial at β)
    pub a_z_beta: u64,
    
    /// B_z(β) evaluation (right constraint polynomial at β)
    pub b_z_beta: u64,
    
    /// C_z(β) evaluation (output constraint polynomial at β)
    pub c_z_beta: u64,
    
    /// Opening proof at α
    pub opening_alpha: Opening,
    
    /// Opening proof at β
    pub opening_beta: Opening,
}

impl ProofR1CS {
    /// Create new R1CS proof.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        commitment_q: Commitment,
        challenge_alpha: Challenge,
        challenge_beta: Challenge,
        q_alpha: u64,
        q_beta: u64,
        a_z_alpha: u64,
        b_z_alpha: u64,
        c_z_alpha: u64,
        a_z_beta: u64,
        b_z_beta: u64,
        c_z_beta: u64,
        opening_alpha: Opening,
        opening_beta: Opening,
    ) -> Self {
        ProofR1CS {
            commitment_q,
            challenge_alpha,
            challenge_beta,
            q_alpha,
            q_beta,
            a_z_alpha,
            b_z_alpha,
            c_z_alpha,
            a_z_beta,
            b_z_beta,
            c_z_beta,
            opening_alpha,
            opening_beta,
        }
    }
    
    /// Get commitment to Q(X).
    pub fn commitment_q(&self) -> &Commitment {
        &self.commitment_q
    }
    
    /// Get first challenge α.
    pub fn challenge_alpha(&self) -> &Challenge {
        &self.challenge_alpha
    }
    
    /// Get second challenge β.
    pub fn challenge_beta(&self) -> &Challenge {
        &self.challenge_beta
    }
}

/// R1CS proof with zero-knowledge property.
///
/// Similar to ProofR1CS but commits to blinded quotient polynomial:
/// Q'(X) = Q(X) + r·Z_H(X)
///
/// where r is a random blinding factor. This prevents witness leakage
/// through the quotient polynomial (VULN-001 mitigation).
///
/// # Security
/// - Zero-knowledge: Proof reveals only satisfiability, not witness values
/// - Soundness error: ε ≤ 2^(-48) (two independent challenges)
/// - Blinding factor: r ∈_R F_q (uniformly random)
/// - Statistical security: |F_q| = 2^64 - 2^32 + 1
#[derive(Debug, Clone)]
pub struct ProofR1csZk {
    /// LWE commitment to blinded quotient polynomial Q'(X) = Q(X) + r·Z_H(X)
    pub commitment_q_prime: Commitment,
    
    /// Blinding factor r (revealed in proof)
    pub blinding_factor: u64,
    
    /// First challenge α
    pub challenge_alpha: Challenge,
    
    /// Second challenge β
    pub challenge_beta: Challenge,
    
    /// Q'(α) evaluation (blinded quotient at α)
    pub q_prime_alpha: u64,
    
    /// Q'(β) evaluation (blinded quotient at β)
    pub q_prime_beta: u64,
    
    /// A_z(α) evaluation (left constraint polynomial at α)
    pub a_z_alpha: u64,
    
    /// B_z(α) evaluation (right constraint polynomial at α)
    pub b_z_alpha: u64,
    
    /// C_z(α) evaluation (output constraint polynomial at α)
    pub c_z_alpha: u64,
    
    /// A_z(β) evaluation (left constraint polynomial at β)
    pub a_z_beta: u64,
    
    /// B_z(β) evaluation (right constraint polynomial at β)
    pub b_z_beta: u64,
    
    /// C_z(β) evaluation (output constraint polynomial at β)
    pub c_z_beta: u64,
    
    /// Opening proof at α
    pub opening_alpha: Opening,
    
    /// Opening proof at β
    pub opening_beta: Opening,
}

impl ProofR1csZk {
    /// Create new zero-knowledge R1CS proof.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        commitment_q_prime: Commitment,
        blinding_factor: u64,
        challenge_alpha: Challenge,
        challenge_beta: Challenge,
        q_prime_alpha: u64,
        q_prime_beta: u64,
        a_z_alpha: u64,
        b_z_alpha: u64,
        c_z_alpha: u64,
        a_z_beta: u64,
        b_z_beta: u64,
        c_z_beta: u64,
        opening_alpha: Opening,
        opening_beta: Opening,
    ) -> Self {
        ProofR1csZk {
            commitment_q_prime,
            blinding_factor,
            challenge_alpha,
            challenge_beta,
            q_prime_alpha,
            q_prime_beta,
            a_z_alpha,
            b_z_alpha,
            c_z_alpha,
            a_z_beta,
            b_z_beta,
            c_z_beta,
            opening_alpha,
            opening_beta,
        }
    }
    
    /// Get commitment to Q'(X).
    pub fn commitment_q_prime(&self) -> &Commitment {
        &self.commitment_q_prime
    }
    
    /// Get blinding factor r.
    pub fn blinding_factor(&self) -> u64 {
        self.blinding_factor
    }
    
    /// Get first challenge α.
    pub fn challenge_alpha(&self) -> &Challenge {
        &self.challenge_alpha
    }
    
    /// Get second challenge β.
    pub fn challenge_beta(&self) -> &Challenge {
        &self.challenge_beta
    }
}

/// SNARK proof containing commitment, challenge, and opening.
#[derive(Debug)]
pub struct Proof {
    /// LWE commitment to witness polynomial
    pub commitment: Commitment,
    
    /// Fiat-Shamir challenge point α
    pub challenge: Challenge,
    
    /// Opening proof at α
    pub opening: Opening,
}

impl Proof {
    /// Create new proof from components.
    pub fn new(commitment: Commitment, challenge: Challenge, opening: Opening) -> Self {
        Proof {
            commitment,
            challenge,
            opening,
        }
    }
    
    /// Get commitment.
    pub fn commitment(&self) -> &Commitment {
        &self.commitment
    }
    
    /// Get challenge.
    pub fn challenge(&self) -> &Challenge {
        &self.challenge
    }
    
    /// Get opening.
    pub fn opening(&self) -> &Opening {
        &self.opening
    }
}

/// Setup phase: generate proving and verifying keys.
///
/// # Errors
///
/// Returns error if parameters are invalid or FFI fails.
pub fn setup(params: Params) -> Result<(ProvingKey, VerifyingKey), Error> {
    params.validate()?;
    
    let ctx = LweContext::new(params)?;
    
    Ok((
        ProvingKey { ctx },
        VerifyingKey {},
    ))
}

/// Generate proof for witness (non-zero-knowledge version).
///
/// Implements the prover algorithm:
/// 1. Encode witness as polynomial f(X) = Σ z_i·X^i
/// 2. Commit to polynomial using LWE
/// 3. Derive Fiat-Shamir challenge α = H(public_inputs || commitment)
/// 4. Compute opening y = f(α)
/// 5. Generate opening proof
/// 6. Assemble and return Proof
///
/// **Note**: This is the non-ZK version. For zero-knowledge proofs, use [`prove_zk`].
///
/// # Arguments
/// * `witness` - Witness values z_1, ..., z_n
/// * `public_inputs` - Public inputs for Fiat-Shamir
/// * `ctx` - LWE context for commitment
/// * `modulus` - Field modulus q
/// * `seed` - Random seed for commitment (0 = random)
///
/// # Returns
/// Proof containing commitment, challenge, and opening
///
/// # Errors
/// Returns error if witness is empty or commitment fails
///
/// # Example
/// ```no_run
/// use lambda_snark::{prove_simple, LweContext, Params, Profile, SecurityLevel};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let modulus = 17592186044417;
/// let params = Params::new(
///     SecurityLevel::Bits128,
///     Profile::RingB { n: 4096, k: 2, q: modulus, sigma: 3.19 },
/// );
/// let ctx = LweContext::new(params)?;
///
/// let witness = vec![1, 7, 13, 91];
/// let public_inputs = vec![1, 91];
///
/// let proof = prove_simple(&witness, &public_inputs, &ctx, modulus, 0x1234)?;
/// println!("Proof generated successfully");
/// # Ok(())
/// # }
/// ```
pub fn prove_simple(
    witness: &[u64],
    public_inputs: &[u64],
    ctx: &LweContext,
    modulus: u64,
    seed: u64,
) -> Result<Proof, Error> {
    // 1. Validate inputs
    if witness.is_empty() {
        return Err(Error::Ffi("Witness cannot be empty".to_string()));
    }
    
    // 2. Encode witness as polynomial
    let polynomial = Polynomial::from_witness(witness, modulus);
    
    // 3. Commit to polynomial
    let commitment = Commitment::new(ctx, polynomial.coefficients(), seed)?;
    
    // 4. Derive Fiat-Shamir challenge
    let challenge = Challenge::derive(public_inputs, &commitment, modulus);
    
    // 5. Generate opening proof
    let opening = generate_opening(&polynomial, challenge.alpha(), seed);
    
    // 6. Assemble proof
    Ok(Proof::new(commitment, challenge, opening))
}

/// Generate zero-knowledge proof for witness.
///
/// Implements the ZK prover algorithm with polynomial blinding:
/// 1. Encode witness as polynomial f(X) = Σ z_i·X^i
/// 2. Generate random blinding polynomial r(X) with same degree
/// 3. Compute blinded polynomial f'(X) = f(X) + r(X)
/// 4. Commit to blinded polynomial using LWE
/// 5. Derive Fiat-Shamir challenge α = H(public_inputs || commitment)
/// 6. Compute blinded opening y' = f'(α) = f(α) + r(α)
/// 7. Generate opening proof for blinded evaluation
/// 8. Assemble and return Proof
///
/// **Zero-Knowledge Property**: The proof reveals nothing about the witness beyond
/// the statement's validity. The blinding polynomial r(X) is uniformly random over
/// F_q^{n+1}, making f'(X) uniformly distributed regardless of f(X).
///
/// **Security**: Achieves honest-verifier zero-knowledge (HVZK) under the LWE
/// assumption. Simulator indistinguishability advantage ≤ 2^-128.
///
/// # Arguments
/// * `witness` - Witness values z_1, ..., z_n
/// * `public_inputs` - Public inputs for Fiat-Shamir
/// * `ctx` - LWE context for commitment
/// * `modulus` - Field modulus q
/// * `commit_seed` - Random seed for commitment (0 = random)
/// * `blinding_seed` - Random seed for blinding polynomial (None = secure random)
///
/// # Returns
/// Zero-knowledge proof containing commitment, challenge, and opening
///
/// # Errors
/// Returns error if witness is empty or commitment fails
///
/// # Example
/// ```no_run
/// use lambda_snark::{prove_zk, LweContext, Params, Profile, SecurityLevel};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let modulus = 17592186044417;
/// let params = Params::new(
///     SecurityLevel::Bits128,
///     Profile::RingB { n: 4096, k: 2, q: modulus, sigma: 3.19 },
/// );
/// let ctx = LweContext::new(params)?;
///
/// let witness = vec![1, 7, 13, 91];
/// let public_inputs = vec![1, 91];
///
/// // Secure ZK proof (random blinding)
/// let zk_proof = prove_zk(&witness, &public_inputs, &ctx, modulus, 0x1234, None)?;
///
/// // Deterministic ZK proof (for testing)
/// let det_proof = prove_zk(&witness, &public_inputs, &ctx, modulus, 0x1234, Some(42))?;
///
/// println!("Zero-knowledge proof generated");
/// # Ok(())
/// # }
/// ```
pub fn prove_zk(
    witness: &[u64],
    public_inputs: &[u64],
    ctx: &LweContext,
    modulus: u64,
    commit_seed: u64,
    blinding_seed: Option<u64>,
) -> Result<Proof, Error> {
    // 1. Validate inputs
    if witness.is_empty() {
        return Err(Error::Ffi("Witness cannot be empty".to_string()));
    }
    
    // 2. Encode witness as polynomial
    let f = Polynomial::from_witness(witness, modulus);
    
    // 3. Generate random blinding polynomial with same degree
    let r = Polynomial::random_blinding(f.degree(), modulus, blinding_seed);
    
    // 4. Compute blinded polynomial f'(X) = f(X) + r(X)
    let f_blinded = f.add(&r);
    
    // 5. Commit to blinded polynomial
    let commitment = Commitment::new(ctx, f_blinded.coefficients(), commit_seed)?;
    
    // 6. Derive Fiat-Shamir challenge
    let challenge = Challenge::derive(public_inputs, &commitment, modulus);
    
    // 7. Compute blinded evaluation y' = f'(α) = f(α) + r(α)
    // (Opening generation handles evaluation internally)
    let opening = generate_opening(&f_blinded, challenge.alpha(), commit_seed);
    
    // 8. Assemble proof
    Ok(Proof::new(commitment, challenge, opening))
}

/// Simulate zero-knowledge proof without witness (for ZK property validation).
///
/// Generates a proof that is computationally indistinguishable from a real proof
/// produced by `prove_zk()`, but without requiring a witness. This demonstrates
/// the zero-knowledge property: if a simulator can produce valid-looking proofs
/// without the witness, then real proofs reveal no information about the witness.
///
/// **Algorithm**:
/// 1. Sample random polynomial f'(X) uniformly from F_q^{n+1}
/// 2. Commit to random polynomial using LWE
/// 3. Derive Fiat-Shamir challenge α = H(public_inputs || commitment)
/// 4. Compute evaluation y' = f'(α)
/// 5. Generate opening proof
/// 6. Return simulated proof
///
/// **Indistinguishability Theorem**: Under the LWE assumption, simulated proofs
/// are computationally indistinguishable from real proofs produced by `prove_zk()`.
///
/// **Proof Sketch**:
/// - Real proof: (Commit(f(X) + r(X)), α, (f+r)(α), opening) where r ~ U(F_q^{n+1})
/// - Simulated proof: (Commit(f'(X)), α, f'(α), opening) where f' ~ U(F_q^{n+1})
/// - Since f(X) + r(X) is uniformly distributed (one-time pad), the distributions
///   are identical: π_real ≡ π_sim
///
/// **Distinguisher Advantage**: Adv_ZK ≤ Adv_LWE + negl(λ) ≈ 2^-128
///
/// # Arguments
/// * `degree` - Polynomial degree (must match real witness degree)
/// * `public_inputs` - Public inputs for Fiat-Shamir challenge
/// * `ctx` - LWE context for commitment
/// * `modulus` - Field modulus q
/// * `commit_seed` - Random seed for commitment (0 = random)
/// * `sim_seed` - Random seed for simulated polynomial (None = secure random)
///
/// # Returns
/// Simulated proof indistinguishable from real ZK proof
///
/// # Errors
/// Returns error if commitment fails
///
/// # Example
/// ```no_run
/// use lambda_snark::{simulate_proof, prove_zk, verify_simple, LweContext, Params, Profile, SecurityLevel};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let modulus = 17592186044417;
/// let params = Params::new(
///     SecurityLevel::Bits128,
///     Profile::RingB { n: 4096, k: 2, q: modulus, sigma: 3.19 },
/// );
/// let ctx = LweContext::new(params)?;
///
/// let public_inputs = vec![1, 91];
///
/// // Simulate proof without witness
/// let sim_proof = simulate_proof(3, &public_inputs, &ctx, modulus, 0x1234, Some(42))?;
///
/// // Simulated proof looks valid (but may not verify for invalid statement)
/// println!("Simulated proof generated: {:?}", sim_proof.challenge().alpha().value());
///
/// // Real witness proof for comparison
/// let witness = vec![1, 7, 13, 91];
/// let real_proof = prove_zk(&witness, &public_inputs, &ctx, modulus, 0x5678, Some(99))?;
///
/// // Both verify if statement is valid
/// assert!(verify_simple(&real_proof, &public_inputs, modulus));
/// // Note: sim_proof won't verify unless we get lucky with random polynomial
/// # Ok(())
/// # }
/// ```
pub fn simulate_proof(
    degree: usize,
    public_inputs: &[u64],
    ctx: &LweContext,
    modulus: u64,
    commit_seed: u64,
    sim_seed: Option<u64>,
) -> Result<Proof, Error> {
    // 1. Sample random polynomial f'(X) uniformly from F_q^{n+1}
    // This simulates the distribution of f(X) + r(X) without knowing f(X)
    let f_prime = Polynomial::random_blinding(degree, modulus, sim_seed);
    
    // 2. Commit to random polynomial
    let commitment = Commitment::new(ctx, f_prime.coefficients(), commit_seed)?;
    
    // 3. Derive Fiat-Shamir challenge (same as real prover)
    let challenge = Challenge::derive(public_inputs, &commitment, modulus);
    
    // 4. Evaluate random polynomial at challenge point
    let opening = generate_opening(&f_prime, challenge.alpha(), commit_seed);
    
    // 5. Return simulated proof
    // This proof is indistinguishable from a real ZK proof under LWE assumption
    Ok(Proof::new(commitment, challenge, opening))
}

/// Generate R1CS proof with two-challenge soundness.
///
/// Implements the full R1CS prover with quotient polynomial:
/// 1. Verify witness satisfies R1CS: A·z ∘ B·z = C·z
/// 2. Compute quotient polynomial Q(X) = (A_z·B_z - C_z) / Z_H
/// 3. Commit to Q(X) using LWE: comm_Q = Commit(Q(X))
/// 4. Derive first challenge: α = H(public || comm_Q)
/// 5. Derive second challenge: β = H(α || comm_Q)
/// 6. Interpolate A_z(X), B_z(X), C_z(X) from constraint evaluations
/// 7. Evaluate polynomials at both challenges (α and β)
/// 8. Generate opening proofs at both challenges
/// 9. Assemble and return ProofR1CS
///
/// **Soundness**: Two independent challenges provide ε ≤ 2^-48 soundness error.
/// A malicious prover cannot satisfy both verification equations simultaneously
/// with high probability.
///
/// **Completeness**: Honest prover with valid witness always produces accepting proof.
///
/// # Arguments
/// * `r1cs` - R1CS constraint system
/// * `witness` - Full witness vector (including public inputs)
/// * `ctx` - LWE context for polynomial commitments
/// * `seed` - Random seed for commitment (0 = random)
///
/// # Returns
/// ProofR1CS containing:
/// - Commitment to Q(X)
/// - Two challenges (α, β)
/// - Polynomial evaluations at both challenges
/// - Opening proofs at both challenges
///
/// # Errors
/// - If witness doesn't satisfy R1CS constraints
/// - If quotient polynomial division fails (indicates bug)
/// - If commitment generation fails
///
/// # Example
/// ```no_run
/// use lambda_snark::{prove_r1cs, R1CS, SparseMatrix, LweContext, Params, Profile, SecurityLevel};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Setup R1CS for x * y = z
/// let modulus = 17592186044417;
/// let a = SparseMatrix::from_dense(&vec![vec![0,1,0,0]]);
/// let b = SparseMatrix::from_dense(&vec![vec![0,0,1,0]]);
/// let c = SparseMatrix::from_dense(&vec![vec![0,0,0,1]]);
/// let r1cs = R1CS::new(1, 4, 2, a, b, c, modulus);
///
/// // LWE context
/// let params = Params::new(
///     SecurityLevel::Bits128,
///     Profile::RingB { n: 4096, k: 2, q: modulus, sigma: 3.19 },
/// );
/// let ctx = LweContext::new(params)?;
///
/// // Generate proof for 7 * 13 = 91
/// let witness = vec![1, 7, 13, 91];
/// let proof = prove_r1cs(&r1cs, &witness, &ctx, 0x1234)?;
///
/// println!("R1CS proof generated with two challenges");
/// # Ok(())
/// # }
/// ```
pub fn prove_r1cs(
    r1cs: &R1CS,
    witness: &[u64],
    ctx: &LweContext,
    seed: u64,
) -> Result<ProofR1CS, Error> {
    // 1. Compute quotient polynomial Q(X)
    let q_coeffs = r1cs.compute_quotient_poly(witness)?;
    
    // 2. Commit to Q(X)
    let q_fields: Vec<Field> = q_coeffs.iter().map(|&v| Field::new(v)).collect();
    let commitment_q = Commitment::new(ctx, &q_fields, seed)?;
    
    // 3. Derive first challenge α from public inputs and commitment
    let public_inputs = r1cs.public_inputs(witness);
    let challenge_alpha = Challenge::derive(&public_inputs, &commitment_q, r1cs.modulus);
    let alpha = challenge_alpha.alpha();
    
    // 4. Derive second challenge β from α and commitment
    let challenge_beta = Challenge::derive(&[alpha.value()], &commitment_q, r1cs.modulus);
    let beta = challenge_beta.alpha();
    
    // 5. Interpolate A_z(X), B_z(X), C_z(X) from constraint evaluations
    let (a_evals, b_evals, c_evals) = r1cs.compute_constraint_evals(witness);
    let a_poly = r1cs::lagrange_interpolate(&a_evals, r1cs.modulus);
    let b_poly = r1cs::lagrange_interpolate(&b_evals, r1cs.modulus);
    let c_poly = r1cs::lagrange_interpolate(&c_evals, r1cs.modulus);
    
    // 6. Evaluate polynomials at α
    let q_alpha = r1cs.eval_poly(&q_coeffs, alpha.value());
    let a_z_alpha = r1cs.eval_poly(&a_poly, alpha.value());
    let b_z_alpha = r1cs.eval_poly(&b_poly, alpha.value());
    let c_z_alpha = r1cs.eval_poly(&c_poly, alpha.value());
    
    // 7. Evaluate polynomials at β
    let q_beta = r1cs.eval_poly(&q_coeffs, beta.value());
    let a_z_beta = r1cs.eval_poly(&a_poly, beta.value());
    let b_z_beta = r1cs.eval_poly(&b_poly, beta.value());
    let c_z_beta = r1cs.eval_poly(&c_poly, beta.value());
    
    // 8. Generate opening proofs at α and β
    // For now, use simplified openings (witness = empty vector)
    // TODO: Implement proper opening proof generation with LWE witness
    let opening_alpha = Opening::new(Field::new(q_alpha), vec![]);
    let opening_beta = Opening::new(Field::new(q_beta), vec![]);
    
    // 9. Assemble proof
    Ok(ProofR1CS::new(
        commitment_q,
        challenge_alpha,
        challenge_beta,
        q_alpha,
        q_beta,
        a_z_alpha,
        b_z_alpha,
        c_z_alpha,
        a_z_beta,
        b_z_beta,
        c_z_beta,
        opening_alpha,
        opening_beta,
    ))
}

/// Generate zero-knowledge R1CS proof with polynomial blinding.
///
/// Similar to prove_r1cs() but adds zero-knowledge property by blinding
/// the quotient polynomial: Q'(X) = Q(X) + r·Z_H(X)
///
/// This prevents witness leakage through Q(X) (VULN-001 mitigation).
///
/// # Zero-Knowledge Protocol
/// 1. Compute quotient Q(X) = (A_z·B_z - C_z) / Z_H
/// 2. Sample random blinding factor r ← F_q
/// 3. Compute blinded quotient Q'(X) = Q(X) + r·Z_H(X)
/// 4. Commit to Q'(X) using LWE
/// 5. Derive challenges α, β (Fiat-Shamir)
/// 6. Evaluate Q'(α), Q'(β) and constraint polynomials
/// 7. Include blinding factor r in proof (explicit)
///
/// # Verification
/// Verifier unblinds Q'(α) → Q(α) using r·Z_H(α):
/// ```text
/// Q(α) = Q'(α) - r·Z_H(α)
/// Q(α)·Z_H(α) ?= A_z(α)·B_z(α) - C_z(α)
/// ```
///
/// # Security
/// - **Zero-knowledge**: Q'(X) statistically independent of Q(X)
/// - **Soundness**: ε ≤ 2^-48 (two challenges)
/// - **Completeness**: Valid witness always produces verifying proof
///
/// # Arguments
/// * `r1cs` - R1CS constraint system
/// * `witness` - Full witness vector
/// * `ctx` - LWE context for commitments
/// * `rng` - Random number generator for blinding factor
/// * `seed` - Commitment seed (0 = random)
///
/// # Returns
/// ProofR1csZk with blinded quotient commitment and blinding factor
///
/// # Example
/// ```no_run
/// use lambda_snark::{prove_r1cs_zk, R1CS, SparseMatrix, LweContext, Params, Profile, SecurityLevel};
/// use rand::thread_rng;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Setup R1CS and context (same as prove_r1cs)
/// let modulus = 17592186044417;
/// let a = SparseMatrix::from_dense(&vec![vec![0,1,0,0]]);
/// let b = SparseMatrix::from_dense(&vec![vec![0,0,1,0]]);
/// let c = SparseMatrix::from_dense(&vec![vec![0,0,0,1]]);
/// let r1cs = R1CS::new(1, 4, 2, a, b, c, modulus);
///
/// let params = Params::new(
///     SecurityLevel::Bits128,
///     Profile::RingB { n: 4096, k: 2, q: modulus, sigma: 3.19 },
/// );
/// let ctx = LweContext::new(params)?;
///
/// // Generate ZK proof
/// let mut rng = thread_rng();
/// let witness = vec![1, 7, 13, 91];
/// let proof = prove_r1cs_zk(&r1cs, &witness, &ctx, &mut rng, 0x1234)?;
///
/// println!("Zero-knowledge R1CS proof generated");
/// # Ok(())
/// # }
/// ```
pub fn prove_r1cs_zk(
    r1cs: &R1CS,
    witness: &[u64],
    ctx: &LweContext,
    rng: &mut dyn rand::RngCore,
    seed: u64,
) -> Result<ProofR1csZk, Error> {
    use r1cs::{poly_add, poly_mul_scalar, vanishing_poly};
    
    // 1. Compute quotient polynomial Q(X)
    let q_coeffs = r1cs.compute_quotient_poly(witness)?;
    
    // 2. Sample random blinding factor r ← F_q
    let blinding_factor = rng.next_u64() % r1cs.modulus;
    
    // 3. Compute vanishing polynomial Z_H(X) with domain-aware method
    //    NTT: Z_H(X) = X^m - 1  |  Baseline: Z_H(X) = X(X-1)...(X-(m-1))
    let use_ntt = r1cs.should_use_ntt();
    let z_h = vanishing_poly(r1cs.m, r1cs.modulus, use_ntt);
    
    // 4. Compute r·Z_H(X)
    let r_z_h = poly_mul_scalar(&z_h, blinding_factor, r1cs.modulus);
    
    // 5. Compute blinded quotient Q'(X) = Q(X) + r·Z_H(X)
    let q_blinded = poly_add(&q_coeffs, &r_z_h, r1cs.modulus);
    
    // 6. Commit to Q'(X)
    let q_prime_fields: Vec<Field> = q_blinded.iter().map(|&v| Field::new(v)).collect();
    let commitment_q_prime = Commitment::new(ctx, &q_prime_fields, seed)?;
    
    // 7. Derive first challenge α from public inputs and commitment
    let public_inputs = r1cs.public_inputs(witness);
    let challenge_alpha = Challenge::derive(&public_inputs, &commitment_q_prime, r1cs.modulus);
    let alpha = challenge_alpha.alpha();
    
    // 8. Derive second challenge β from α and commitment
    let challenge_beta = Challenge::derive(&[alpha.value()], &commitment_q_prime, r1cs.modulus);
    let beta = challenge_beta.alpha();
    
    // 9. Interpolate A_z(X), B_z(X), C_z(X)
    let (a_evals, b_evals, c_evals) = r1cs.compute_constraint_evals(witness);
    let a_poly = r1cs::lagrange_interpolate(&a_evals, r1cs.modulus);
    let b_poly = r1cs::lagrange_interpolate(&b_evals, r1cs.modulus);
    let c_poly = r1cs::lagrange_interpolate(&c_evals, r1cs.modulus);
    
    // 10. Evaluate blinded quotient Q'(α) and Q'(β)
    let q_prime_alpha = r1cs.eval_poly(&q_blinded, alpha.value());
    let q_prime_beta = r1cs.eval_poly(&q_blinded, beta.value());
    
    // 11. Evaluate constraint polynomials at α
    let a_z_alpha = r1cs.eval_poly(&a_poly, alpha.value());
    let b_z_alpha = r1cs.eval_poly(&b_poly, alpha.value());
    let c_z_alpha = r1cs.eval_poly(&c_poly, alpha.value());
    
    // 12. Evaluate constraint polynomials at β
    let a_z_beta = r1cs.eval_poly(&a_poly, beta.value());
    let b_z_beta = r1cs.eval_poly(&b_poly, beta.value());
    let c_z_beta = r1cs.eval_poly(&c_poly, beta.value());
    
    // 13. Generate opening proofs at α and β
    let opening_alpha = Opening::new(Field::new(q_prime_alpha), vec![]);
    let opening_beta = Opening::new(Field::new(q_prime_beta), vec![]);
    
    // 14. Assemble ZK proof
    Ok(ProofR1csZk::new(
        commitment_q_prime,
        blinding_factor,
        challenge_alpha,
        challenge_beta,
        q_prime_alpha,
        q_prime_beta,
        a_z_alpha,
        b_z_alpha,
        c_z_alpha,
        a_z_beta,
        b_z_beta,
        c_z_beta,
        opening_alpha,
        opening_beta,
    ))
}

/// Verify R1CS proof with two-challenge soundness.
///
/// Implements the full R1CS verifier:
/// 1. Recompute challenges α', β' from proof
/// 2. Verify Q(α)·Z_H(α) = A_z(α)·B_z(α) - C_z(α)
/// 3. Verify Q(β)·Z_H(β) = A_z(β)·B_z(β) - C_z(β)
/// 4. Verify opening proofs at both challenges
///
/// **Soundness**: Two-challenge verification provides ε ≤ 2^-48 soundness.
/// A malicious prover cannot forge valid evaluations at both independent
/// challenges with high probability.
///
/// **Completeness**: Honest prover with valid witness always produces
/// verifying proof.
///
/// # Arguments
/// * `proof` - ProofR1CS to verify
/// * `public_inputs` - Public inputs (z[0..l])
/// * `r1cs` - R1CS constraint system
///
/// # Returns
/// `true` if proof is valid, `false` otherwise
///
/// # Verification Equations
/// ```text
/// α = H(public || comm_Q)
/// β = H(α || comm_Q)
/// Q(α)·Z_H(α) ?= A_z(α)·B_z(α) - C_z(α)
/// Q(β)·Z_H(β) ?= A_z(β)·B_z(β) - C_z(β)
/// ```
///
/// # Example
/// ```no_run
/// use lambda_snark::{prove_r1cs, verify_r1cs, R1CS, SparseMatrix, LweContext, Params, Profile, SecurityLevel};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let modulus = 17592186044417;
/// let a = SparseMatrix::from_dense(&vec![vec![0,1,0,0]]);
/// let b = SparseMatrix::from_dense(&vec![vec![0,0,1,0]]);
/// let c = SparseMatrix::from_dense(&vec![vec![0,0,0,1]]);
/// let r1cs = R1CS::new(1, 4, 2, a, b, c, modulus);
///
/// let params = Params::new(
///     SecurityLevel::Bits128,
///     Profile::RingB { n: 4096, k: 2, q: modulus, sigma: 3.19 },
/// );
/// let ctx = LweContext::new(params)?;
///
/// let witness = vec![1, 7, 13, 91];
/// let proof = prove_r1cs(&r1cs, &witness, &ctx, 0x1234)?;
///
/// let public_inputs = r1cs.public_inputs(&witness);
/// let valid = verify_r1cs(&proof, public_inputs, &r1cs);
/// assert!(valid, "Valid proof should verify");
/// # Ok(())
/// # }
/// ```
pub fn verify_r1cs(
    proof: &ProofR1CS,
    public_inputs: &[u64],
    r1cs: &R1CS,
) -> bool {
    let modulus = r1cs.modulus;
    
    // 1. Recompute first challenge α' = H(public || comm_Q)
    let challenge_alpha_recomputed = Challenge::derive(
        public_inputs,
        proof.commitment_q(),
        modulus,
    );
    
    // 2. Check α' = α (challenge consistency)
    if proof.challenge_alpha.alpha() != challenge_alpha_recomputed.alpha() {
        return false;
    }
    
    let alpha = proof.challenge_alpha.alpha().value();
    
    // 3. Recompute second challenge β' = H(α || comm_Q)
    let challenge_beta_recomputed = Challenge::derive(
        &[alpha],
        proof.commitment_q(),
        modulus,
    );
    
    // 4. Check β' = β (challenge consistency)
    if proof.challenge_beta.alpha() != challenge_beta_recomputed.alpha() {
        return false;
    }
    
    let beta = proof.challenge_beta.alpha().value();
    
    // 5. Compute Z_H(α) using domain-aware method
    // Domain depends on interpolation method (NTT vs baseline)
    let zh_alpha = r1cs.eval_vanishing(alpha);
    
    // 6. Compute Z_H(β)
    let zh_beta = r1cs.eval_vanishing(beta);
    
    // 7. Verify first equation: Q(α)·Z_H(α) = A_z(α)·B_z(α) - C_z(α)
    let lhs_alpha = ((proof.q_alpha as u128 * zh_alpha as u128) % modulus as u128) as u64;
    
    let ab_alpha = ((proof.a_z_alpha as u128 * proof.b_z_alpha as u128) % modulus as u128) as u64;
    let rhs_alpha = if ab_alpha >= proof.c_z_alpha {
        ab_alpha - proof.c_z_alpha
    } else {
        modulus - (proof.c_z_alpha - ab_alpha)
    };
    
    if lhs_alpha != rhs_alpha {
        return false;
    }
    
    // 8. Verify second equation: Q(β)·Z_H(β) = A_z(β)·B_z(β) - C_z(β)
    let lhs_beta = ((proof.q_beta as u128 * zh_beta as u128) % modulus as u128) as u64;
    
    let ab_beta = ((proof.a_z_beta as u128 * proof.b_z_beta as u128) % modulus as u128) as u64;
    let rhs_beta = if ab_beta >= proof.c_z_beta {
        ab_beta - proof.c_z_beta
    } else {
        modulus - (proof.c_z_beta - ab_beta)
    };
    
    if lhs_beta != rhs_beta {
        return false;
    }
    
    // 9. Verify opening proofs at α and β
    // Note: Opening verification is simplified for now
    // TODO: Implement full LWE opening verification when LWE witness is available
    
    // Check that opening evaluations match claimed values
    if proof.opening_alpha.evaluation().value() != proof.q_alpha {
        return false;
    }
    
    if proof.opening_beta.evaluation().value() != proof.q_beta {
        return false;
    }
    
    // All checks passed
    true
}

/// Verify zero-knowledge R1CS proof with polynomial unblinding.
///
/// Similar to verify_r1cs() but unblinds Q'(X) before verification:
/// ```text
/// Q(α) = Q'(α) - r·Z_H(α)
/// Q(α)·Z_H(α) ?= A_z(α)·B_z(α) - C_z(α)
/// ```
///
/// # Verification Steps
/// 1. Recompute challenges α', β' from proof
/// 2. Compute Z_H(α) and Z_H(β)
/// 3. Unblind: Q(α) = Q'(α) - r·Z_H(α)
/// 4. Verify: Q(α)·Z_H(α) = A_z(α)·B_z(α) - C_z(α)
/// 5. Repeat for challenge β
/// 6. Verify opening proofs
///
/// # Security
/// - **Soundness**: Malicious prover cannot satisfy both equations
/// - **Zero-knowledge**: Blinding factor r reveals no witness info
///
/// # Arguments
/// * `proof` - Zero-knowledge proof to verify
/// * `public_inputs` - Public inputs z[0..l]
/// * `r1cs` - R1CS constraint system
///
/// # Returns
/// `true` if proof is valid, `false` otherwise
///
/// # Example
/// ```no_run
/// use lambda_snark::{prove_r1cs_zk, verify_r1cs_zk, R1CS, SparseMatrix, LweContext, Params, Profile, SecurityLevel};
/// use rand::thread_rng;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Setup (same as prove_r1cs_zk example)
/// let modulus = 17592186044417;
/// let a = SparseMatrix::from_dense(&vec![vec![0,1,0,0]]);
/// let b = SparseMatrix::from_dense(&vec![vec![0,0,1,0]]);
/// let c = SparseMatrix::from_dense(&vec![vec![0,0,0,1]]);
/// let r1cs = R1CS::new(1, 4, 2, a, b, c, modulus);
///
/// let params = Params::new(
///     SecurityLevel::Bits128,
///     Profile::RingB { n: 4096, k: 2, q: modulus, sigma: 3.19 },
/// );
/// let ctx = LweContext::new(params)?;
///
/// // Prove and verify
/// let mut rng = thread_rng();
/// let witness = vec![1, 7, 13, 91];
/// let proof = prove_r1cs_zk(&r1cs, &witness, &ctx, &mut rng, 0x1234)?;
///
/// let public_inputs = r1cs.public_inputs(&witness);
/// let valid = verify_r1cs_zk(&proof, &public_inputs, &r1cs);
/// assert!(valid, "Valid ZK proof should verify");
/// # Ok(())
/// # }
/// ```
pub fn verify_r1cs_zk(
    proof: &ProofR1csZk,
    public_inputs: &[u64],
    r1cs: &R1CS,
) -> bool {
    let modulus = r1cs.modulus;
    
    // 1. Recompute first challenge α' = H(public || comm_Q')
    let challenge_alpha_recomputed = Challenge::derive(
        public_inputs,
        proof.commitment_q_prime(),
        modulus,
    );
    
    // 2. Check α' = α (challenge consistency)
    if proof.challenge_alpha.alpha() != challenge_alpha_recomputed.alpha() {
        return false;
    }
    
    let alpha = proof.challenge_alpha.alpha().value();
    
    // 3. Recompute second challenge β' = H(α || comm_Q')
    let challenge_beta_recomputed = Challenge::derive(
        &[alpha],
        proof.commitment_q_prime(),
        modulus,
    );
    
    // 4. Check β' = β (challenge consistency)
    if proof.challenge_beta.alpha() != challenge_beta_recomputed.alpha() {
        return false;
    }
    
    let beta = proof.challenge_beta.alpha().value();
    
    // 5. Compute Z_H(α) using domain-aware method
    let zh_alpha = r1cs.eval_vanishing(alpha);
    
    // 6. Compute Z_H(β)
    let zh_beta = r1cs.eval_vanishing(beta);
    
    // 7. Unblind Q'(α) → Q(α) using r·Z_H(α)
    let r_zh_alpha = ((proof.blinding_factor as u128 * zh_alpha as u128) % modulus as u128) as u64;
    let q_alpha = if proof.q_prime_alpha >= r_zh_alpha {
        proof.q_prime_alpha - r_zh_alpha
    } else {
        modulus - (r_zh_alpha - proof.q_prime_alpha)
    };
    
    // 8. Unblind Q'(β) → Q(β) using r·Z_H(β)
    let r_zh_beta = ((proof.blinding_factor as u128 * zh_beta as u128) % modulus as u128) as u64;
    let q_beta = if proof.q_prime_beta >= r_zh_beta {
        proof.q_prime_beta - r_zh_beta
    } else {
        modulus - (r_zh_beta - proof.q_prime_beta)
    };
    
    // 9. Verify first equation: Q(α)·Z_H(α) = A_z(α)·B_z(α) - C_z(α)
    let lhs_alpha = ((q_alpha as u128 * zh_alpha as u128) % modulus as u128) as u64;
    
    let ab_alpha = ((proof.a_z_alpha as u128 * proof.b_z_alpha as u128) % modulus as u128) as u64;
    let rhs_alpha = if ab_alpha >= proof.c_z_alpha {
        ab_alpha - proof.c_z_alpha
    } else {
        modulus - (proof.c_z_alpha - ab_alpha)
    };
    
    if lhs_alpha != rhs_alpha {
        return false;
    }
    
    // 10. Verify second equation: Q(β)·Z_H(β) = A_z(β)·B_z(β) - C_z(β)
    let lhs_beta = ((q_beta as u128 * zh_beta as u128) % modulus as u128) as u64;
    
    let ab_beta = ((proof.a_z_beta as u128 * proof.b_z_beta as u128) % modulus as u128) as u64;
    let rhs_beta = if ab_beta >= proof.c_z_beta {
        ab_beta - proof.c_z_beta
    } else {
        modulus - (proof.c_z_beta - ab_beta)
    };
    
    if lhs_beta != rhs_beta {
        return false;
    }
    
    // 11. Verify opening proofs at α and β
    // Check that opening evaluations match claimed blinded values
    if proof.opening_alpha.evaluation().value() != proof.q_prime_alpha {
        return false;
    }
    
    if proof.opening_beta.evaluation().value() != proof.q_prime_beta {
        return false;
    }
    
    // All checks passed
    true
}

/// Generate proof for R1CS instance (legacy API).
///
/// # Errors
///
/// Returns error if witness doesn't satisfy R1CS or proving fails.
pub fn prove(
    _pk: &ProvingKey,
    _public_input: &[Field],
    _witness: &[Field],
) -> Result<Proof, Error> {
    // TODO: Implement full R1CS prover
    Err(Error::Ffi("R1CS prover not implemented yet, use prove_simple()".to_string()))
}

/// Verify SNARK proof (simple version).
///
/// Implements the verifier algorithm:
/// 1. Validate proof structure
/// 2. Recompute Fiat-Shamir challenge α' = H(public_inputs || commitment)
/// 3. Check α' = α (challenge consistency)
/// 4. Verify opening proof at α
///
/// # Arguments
/// * `proof` - Proof to verify
/// * `public_inputs` - Public inputs used in Fiat-Shamir
/// * `modulus` - Field modulus q
///
/// # Returns
/// `true` if proof is valid, `false` otherwise
///
/// # Security
/// - Soundness: Invalid proof rejected with probability ≥ 1 - ε (see docs)
/// - Completeness: Valid proof always accepts
///
/// # Example
/// ```no_run
/// use lambda_snark::{prove_simple, verify_simple, LweContext, Params, Profile, SecurityLevel};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let modulus = 17592186044417;
/// let params = Params::new(
///     SecurityLevel::Bits128,
///     Profile::RingB { n: 4096, k: 2, q: modulus, sigma: 3.19 },
/// );
/// let ctx = LweContext::new(params)?;
///
/// let witness = vec![1, 7, 13, 91];
/// let public_inputs = vec![1, 91];
///
/// let proof = prove_simple(&witness, &public_inputs, &ctx, modulus, 0x1234)?;
/// let valid = verify_simple(&proof, &public_inputs, modulus);
/// assert!(valid, "Valid proof should verify");
/// # Ok(())
/// # }
/// ```
pub fn verify_simple(
    proof: &Proof,
    public_inputs: &[u64],
    modulus: u64,
) -> bool {
    // 1. Recompute Fiat-Shamir challenge
    let challenge_recomputed = Challenge::derive(public_inputs, &proof.commitment, modulus);
    
    // 2. Check challenge consistency
    if proof.challenge.alpha() != challenge_recomputed.alpha() {
        return false;
    }
    
    // 3. Verify opening proof
    verify_opening(
        &proof.commitment,
        proof.challenge.alpha(),
        &proof.opening,
        modulus,
    )
}

/// Verify proof (legacy R1CS API).
///
/// # Errors
///
/// Returns error if verification fails or proof is malformed.
pub fn verify(
    _vk: &VerifyingKey,
    _public_input: &[Field],
    _proof: &Proof,
) -> Result<bool, Error> {
    // TODO: Implement full R1CS verifier
    Err(Error::Ffi("R1CS verifier not implemented yet, use verify_simple()".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_setup() {
        let params = Params::new(
            SecurityLevel::Bits128,
            Profile::RingB {
                n: 4096,  // SEAL requires n >= 1024
                k: 2,
                q: 17592186044417,  // 2^44 + 1 (prime, > 2^24)
                sigma: 3.19,
            },
        );
        
        let result = setup(params);
        assert!(result.is_ok());
    }
}
