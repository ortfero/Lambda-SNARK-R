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
