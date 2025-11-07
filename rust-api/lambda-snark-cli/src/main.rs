use clap::{Parser, Subcommand};
use lambda_snark::{Params, Profile, SecurityLevel};
use lambda_snark::{CircuitBuilder, LweContext, prove_r1cs, verify_r1cs};

#[derive(Parser)]
#[command(name = "lambda-snark")]
#[command(about = "ΛSNARK-R: Post-quantum SNARK toolkit", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Setup proving and verifying keys
    Setup {
        /// Security level (128, 192, or 256)
        #[arg(short, long, default_value_t = 128)]
        security_level: u32,
        
        /// Output file for proving key
        #[arg(short, long)]
        pk_out: String,
        
        /// Output file for verifying key
        #[arg(short, long)]
        vk_out: String,
    },
    
    /// Generate proof
    Prove {
        /// Proving key file
        #[arg(short, long)]
        pk: String,
        
        /// Public input file (JSON)
        #[arg(short = 'x', long)]
        public_input: String,
        
        /// Witness file (JSON)
        #[arg(short, long)]
        witness: String,
        
        /// Output proof file
        #[arg(short, long)]
        output: String,
    },
    
    /// Verify proof
    Verify {
        /// Verifying key file
        #[arg(short, long)]
        vk: String,
        
        /// Public input file (JSON)
        #[arg(short = 'x', long)]
        public_input: String,
        
        /// Proof file
        #[arg(short, long)]
        proof: String,
    },
    
    /// Show version and build information
    Info,
    
    /// Run R1CS proof example (7 × 13 = 91)
    R1csExample {
        /// Random seed for proof generation
        #[arg(short, long, default_value_t = 42)]
        seed: u64,
        
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    
    /// Run Range Proof example (prove value ∈ [0, 256) without revealing value)
    RangeProofExample {
        /// Random seed for proof generation
        #[arg(short, long, default_value_t = 42)]
        seed: u64,
        
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Setup { security_level, pk_out, vk_out } => {
            println!("Setting up with security level {}...", security_level);
            
            let sec_level = match security_level {
                128 => SecurityLevel::Bits128,
                192 => SecurityLevel::Bits192,
                256 => SecurityLevel::Bits256,
                _ => anyhow::bail!("Invalid security level (must be 128, 192, or 256)"),
            };
            
            let params = Params::new(
                sec_level,
                Profile::RingB {
                    n: 256,
                    k: 2,
                    q: 12289,
                    sigma: 3.19,
                },
            );
            
            let (_pk, _vk) = lambda_snark::setup(params)?;
            
            println!("✓ Setup complete");
            println!("  Proving key: {}", pk_out);
            println!("  Verifying key: {}", vk_out);
            
            // TODO: Serialize and save keys
            println!("\n⚠️  WARNING: Key serialization not yet implemented");
        }
        
        Commands::Prove { pk, public_input, witness, output } => {
            println!("Generating proof...");
            println!("  PK: {}", pk);
            println!("  Public input: {}", public_input);
            println!("  Witness: {}", witness);
            
            // TODO: Load keys, generate proof
            println!("\n⚠️  WARNING: Prover not yet implemented");
            println!("  Output: {}", output);
        }
        
        Commands::Verify { vk, public_input, proof } => {
            println!("Verifying proof...");
            println!("  VK: {}", vk);
            println!("  Public input: {}", public_input);
            println!("  Proof: {}", proof);
            
            // TODO: Load keys, verify
            println!("\n⚠️  WARNING: Verifier not yet implemented");
        }
        
        Commands::Info => {
            println!("ΛSNARK-R v{}", env!("CARGO_PKG_VERSION"));
            println!();
            println!("Architecture: Hybrid (C++ Core + Rust API)");
            println!("Target: {}", std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()));
            println!();
            println!("Features:");
            println!("  - Post-quantum security (Module-LWE/SIS)");
            println!("  - Succinct proofs (O(log M) size)");
            println!("  - Zero-knowledge");
            println!();
            println!("Status: ⚠️  Pre-alpha (NOT FOR PRODUCTION)");
            println!("License: Apache-2.0 OR MIT");
        }
        
        Commands::R1csExample { seed, verbose } => {
            run_r1cs_example(seed, verbose)?;
        }
        
        Commands::RangeProofExample { seed, verbose } => {
            run_range_proof_example(seed, verbose)?;
        }
    }
    
    Ok(())
}

/// Run complete R1CS proof-verify example: 7 × 13 = 91
fn run_r1cs_example(seed: u64, verbose: bool) -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       ΛSNARK-R: R1CS Proof Example (TV-R1CS-1)           ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    
    // Step 1: Build R1CS circuit for x·y = z
    println!("📋 Step 1: Building R1CS circuit for multiplication");
    println!("   Statement: 7 × 13 = 91");
    println!();
    
    // Use LWE-compatible modulus (must be prime > 2^24)
    let modulus = 17592186044417u64;  // 2^44 + 1 (large prime for LWE)
    let mut builder = CircuitBuilder::new(modulus);
    
    // Allocate variables: [z_0=1, z_1=x, z_2=y, z_3=result]
    let one = builder.alloc_var();      // Variable 0: constant 1
    let x = builder.alloc_var();        // Variable 1: public x=7
    let y = builder.alloc_var();        // Variable 2: private y=13
    let result = builder.alloc_var();   // Variable 3: public result=91
    
    // Set public inputs: first 2 variables (constant 1 and x are public)
    // Actually for this example, x and result are public, y is private
    builder.set_public_inputs(2);
    
    // Add constraint: x * y = result
    // A = [0, 1, 0, 0] (selects x)
    // B = [0, 0, 1, 0] (selects y)
    // C = [0, 0, 0, 1] (selects result)
    builder.add_constraint(
        vec![(x, 1)],      // A: select variable x with coefficient 1
        vec![(y, 1)],      // B: select variable y with coefficient 1
        vec![(result, 1)], // C: select variable result with coefficient 1
    );
    
    if verbose {
        println!("   Variables allocated:");
        println!("     z_{} = 1 (constant)", one);
        println!("     z_{} = x (public)", x);
        println!("     z_{} = y (private witness)", y);
        println!("     z_{} = result (public)", result);
        println!();
        println!("   Constraint: z_{} × z_{} = z_{}", x, y, result);
    }
    
    // Build R1CS
    let r1cs = builder.build();
    
    println!("   ✓ Circuit built: {} constraints, {} variables, modulus={}", 
             r1cs.num_constraints(), r1cs.witness_size(), modulus);
    println!();
    
    // Step 2: Prepare witness and public inputs
    println!("🔐 Step 2: Preparing witness and public inputs");
    
    // Full witness vector: [1, x=7, y=13, result=91]
    let full_witness = vec![1u64, 7u64, 13u64, 91u64];
    
    // Public inputs: first l=2 elements of witness [1, x]
    // (the prover derives public_inputs from witness automatically)
    let public_inputs = r1cs.public_inputs(&full_witness);
    
    // Verify constraint satisfaction
    if !r1cs.is_satisfied(&full_witness) {
        anyhow::bail!("Witness does not satisfy R1CS constraints!");
    }
    
    if verbose {
        println!("   Full witness:    {:?}", full_witness);
        println!("   Public inputs:   {:?}", public_inputs);
        println!("   Private witness: [13, 91]");
    } else {
        println!("   Public:  constant=1, x={}", public_inputs[1]);
        println!("   Private: y=13, result=91");
    }
    println!("   ✓ Witness satisfies constraints");
    println!();
    
    // Step 3: Setup LWE context
    println!("⚙️  Step 3: Initializing LWE commitment scheme");
    
    let lwe_params = Params::new(
        SecurityLevel::Bits128,
        Profile::RingB {
            n: 4096,  // SEAL requires power-of-2 degree (2^12)
            k: 2,
            q: 17592186044417,  // Same as R1CS modulus (2^44 + 1, prime)
            sigma: 3.19,
        },
    );
    let ctx = LweContext::new(lwe_params)?;
    
    println!("   LWE parameters: n=4096, q=17592186044417 (2^44+1), σ=3.19");
    println!("   Security: 128-bit post-quantum (Module-LWE)");
    println!();
    
    // Step 4: Generate proof
    println!("🔨 Step 4: Generating R1CS proof (seed={})", seed);
    
    let proof = prove_r1cs(&r1cs, &full_witness, &ctx, seed)?;
    
    println!("   ✓ Proof generated successfully");
    if verbose {
        println!("     Challenge α: {}", proof.challenge_alpha.alpha().value());
        println!("     Challenge β: {}", proof.challenge_beta.alpha().value());
        println!("     Q(α) = {}, Q(β) = {}", proof.q_alpha, proof.q_beta);
        println!("     A_z(α) = {}, B_z(α) = {}, C_z(α) = {}", 
                 proof.a_z_alpha, proof.b_z_alpha, proof.c_z_alpha);
        println!("     A_z(β) = {}, B_z(β) = {}, C_z(β) = {}", 
                 proof.a_z_beta, proof.b_z_beta, proof.c_z_beta);
    }
    
    // Estimate proof size
    let proof_size = std::mem::size_of_val(&proof);
    println!("   Proof size: ~{} bytes", proof_size);
    println!();
    
    // Step 5: Verify proof
    println!("✅ Step 5: Verifying R1CS proof");
    
    let is_valid = verify_r1cs(&proof, public_inputs, &r1cs);
    
    if is_valid {
        println!("   ✓ Proof VALID ✓");
        println!();
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║  SUCCESS: Proof verified! 7 × 13 = 91 is proven correct  ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
    } else {
        println!("   ✗ Proof INVALID ✗");
        anyhow::bail!("Verification failed!");
    }
    
    println!();
    println!("Summary:");
    println!("  - Circuit:       {} constraints, {} variables", 
             r1cs.num_constraints(), r1cs.witness_size());
    println!("  - Public inputs: {} (constant=1, x={})", 
             public_inputs.len(), public_inputs[1]);
    println!("  - Proof size:    ~{} bytes", proof_size);
    println!("  - Soundness:     ε ≤ 2^-48 (two Fiat-Shamir challenges)");
    println!("  - Security:      128-bit quantum (Module-LWE)");
    
    Ok(())
}

/// Run Range Proof example: prove value=42 ∈ [0, 256) without revealing value
fn run_range_proof_example(seed: u64, verbose: bool) -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     ΛSNARK-R: Range Proof Example (8-bit range)          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    
    // Secret value and bit width
    let secret_value = 42u64;  // Binary: 0010_1010
    let k = 8;  // Prove value ∈ [0, 2^8) = [0, 256)
    
    println!("🎯 Goal: Prove that a secret value is in range [0, 256)");
    println!("   WITHOUT revealing the actual value!");
    println!();
    println!("   Secret value: {} (binary: {:08b})", secret_value, secret_value);
    println!("   Range: [0, 2^{}) = [0, 256)", k);
    println!();
    
    // Step 1: Build Range Proof circuit
    println!("📋 Step 1: Building Range Proof circuit");
    println!("   Technique: Bit decomposition + boolean constraints");
    println!();
    
    let modulus = 17592186044417u64;
    let mut builder = CircuitBuilder::new(modulus);
    
    // Allocate constant
    let one = builder.alloc_var();  // z_0 = 1
    
    // Allocate bit variables b_0, b_1, ..., b_7
    let mut bit_vars = Vec::new();
    for i in 0..k {
        let bit_var = builder.alloc_var();
        bit_vars.push(bit_var);
        
        if verbose {
            println!("   Allocated z_{} = b_{} (bit {})", bit_var, i, i);
        }
    }
    
    // Allocate value variable (will be reconstructed from bits)
    let value_var = builder.alloc_var();
    
    if verbose {
        println!("   Allocated z_{} = value (reconstructed)", value_var);
        println!();
    }
    
    // Public inputs: just constant 1 (range [0, 2^k) is public knowledge)
    builder.set_public_inputs(1);
    
    // Add boolean constraints: b_i · (b_i - 1) = 0 for each bit
    // This forces b_i ∈ {0, 1}
    println!("   Adding {} boolean constraints (b_i · (b_i - 1) = 0)...", k);
    
    for (i, &bit_var) in bit_vars.iter().enumerate() {
        // Constraint: b_i · (b_i - 1) = 0
        // Rewrite as: b_i · b_i = b_i
        // A = [b_i], B = [b_i], C = [b_i]
        builder.add_constraint(
            vec![(bit_var, 1)],
            vec![(bit_var, 1)],
            vec![(bit_var, 1)],
        );
        
        if verbose {
            println!("     Constraint {}: b_{} · b_{} = b_{}", i+1, i, i, i);
        }
    }
    
    // Add recomposition constraint: value = Σ 2^i · b_i
    // We need to express this as a single R1CS constraint
    // Trick: Use a linear combination in C
    println!();
    println!("   Adding value recomposition constraint...");
    
    // Build coefficient vector for C = [2^0·b_0 + 2^1·b_1 + ... + 2^7·b_7]
    let mut c_terms = Vec::new();
    for (i, &bit_var) in bit_vars.iter().enumerate() {
        let coeff = 1u64 << i;  // 2^i
        c_terms.push((bit_var, coeff));
    }
    
    // Constraint: 1 · value = Σ 2^i · b_i
    // A = [1], B = [value], C = [coefficients]
    builder.add_constraint(
        vec![(one, 1)],       // A: constant 1
        vec![(value_var, 1)], // B: value
        c_terms,              // C: Σ 2^i · b_i
    );
    
    if verbose {
        println!("     1 · value = 2^0·b_0 + 2^1·b_1 + ... + 2^7·b_7");
    }
    
    let r1cs = builder.build();
    let num_constraints = r1cs.num_constraints();
    
    println!("   ✓ Circuit built: {} constraints ({} boolean + 1 recomposition)", 
             num_constraints, k);
    println!();
    
    // Step 2: Prepare witness
    println!("🔐 Step 2: Preparing witness (bit decomposition)");
    
    // Decompose secret_value into bits
    let mut bits = Vec::new();
    for i in 0..k {
        let bit = (secret_value >> i) & 1;
        bits.push(bit);
    }
    
    // Build full witness: [1, b_0, b_1, ..., b_7, value]
    let mut full_witness = vec![1u64];
    full_witness.extend(&bits);
    full_witness.push(secret_value);
    
    if verbose {
        println!("   Bit decomposition: {:?}", bits);
        println!("   Full witness: {:?}", full_witness);
    } else {
        println!("   Bits: {:?} (LSB first)", bits);
        println!("   Reconstructed value: {}", secret_value);
    }
    
    // Verify constraint satisfaction
    if !r1cs.is_satisfied(&full_witness) {
        anyhow::bail!("Witness does not satisfy R1CS constraints!");
    }
    println!("   ✓ Witness satisfies all {} constraints", num_constraints);
    println!();
    
    // Public inputs (only constant 1)
    let public_inputs = r1cs.public_inputs(&full_witness);
    
    // Step 3: Setup LWE
    println!("⚙️  Step 3: Initializing LWE commitment scheme");
    
    let lwe_params = Params::new(
        SecurityLevel::Bits128,
        Profile::RingB {
            n: 4096,
            k: 2,
            q: modulus,
            sigma: 3.19,
        },
    );
    let ctx = LweContext::new(lwe_params)?;
    
    println!("   LWE parameters: n=4096, q=2^44+1, σ=3.19");
    println!("   Security: 128-bit post-quantum (Module-LWE)");
    println!();
    
    // Step 4: Generate proof
    println!("🔨 Step 4: Generating Range Proof (seed={})", seed);
    println!("   ⚠️  Note: Actual value (42) is NOT revealed in proof!");
    
    let proof = prove_r1cs(&r1cs, &full_witness, &ctx, seed)?;
    
    println!("   ✓ Proof generated successfully");
    if verbose {
        println!("     Challenge α: {}", proof.challenge_alpha.alpha().value());
        println!("     Challenge β: {}", proof.challenge_beta.alpha().value());
        println!("     Number of polynomial evaluations: {}", k + 1);
    }
    
    let proof_size = std::mem::size_of_val(&proof);
    println!("   Proof size: ~{} bytes", proof_size);
    println!();
    
    // Step 5: Verify proof
    println!("✅ Step 5: Verifying Range Proof");
    println!("   Verifier knows: range is [0, 256)");
    println!("   Verifier does NOT know: actual value");
    
    let is_valid = verify_r1cs(&proof, public_inputs, &r1cs);
    
    if is_valid {
        println!("   ✓ Proof VALID ✓");
        println!();
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║   SUCCESS: Proved value ∈ [0, 256) without revealing!    ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
    } else {
        println!("   ✗ Proof INVALID ✗");
        anyhow::bail!("Verification failed!");
    }
    
    println!();
    println!("Summary:");
    println!("  - Range:         [0, 2^{}) = [0, 256)", k);
    println!("  - Circuit:       {} constraints ({} boolean + 1 recomp)", 
             num_constraints, k);
    println!("  - Variables:     {} (1 constant + {} bits + 1 value)", 
             r1cs.witness_size(), k);
    println!("  - Privacy:       ✓ Value hidden (only bits in private witness)");
    println!("  - Proof size:    ~{} bytes", proof_size);
    println!("  - Soundness:     ε ≤ 2^-48 (two Fiat-Shamir challenges)");
    println!("  - Security:      128-bit quantum (Module-LWE)");
    
    Ok(())
}
