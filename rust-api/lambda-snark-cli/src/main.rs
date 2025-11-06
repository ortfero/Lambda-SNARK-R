use clap::{Parser, Subcommand};
use lambda_snark::{Params, Profile, SecurityLevel};

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
            println!("C++ Compiler: {}", env!("CARGO_CFG_TARGET_ENV"));
            println!("Rust Version: {}", env!("CARGO_PKG_RUST_VERSION"));
            println!();
            println!("Features:");
            println!("  - Post-quantum security (Module-LWE/SIS)");
            println!("  - Succinct proofs (O(log M) size)");
            println!("  - Zero-knowledge");
            println!();
            println!("Status: ⚠️  Pre-alpha (NOT FOR PRODUCTION)");
            println!("License: Apache-2.0 OR MIT");
        }
    }
    
    Ok(())
}
