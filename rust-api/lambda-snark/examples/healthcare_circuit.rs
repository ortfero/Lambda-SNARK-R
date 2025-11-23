#[path = "healthcare/shared.rs"]
mod shared;

pub use shared::{
    artifacts_dir, build_healthcare_circuit, high_risk_witness, low_risk_witness,
    run_example,
};

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    shared::run_example()
}
