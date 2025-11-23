//! Dump quotient polynomial coefficients for the healthcare example.

#[path = "healthcare/shared.rs"]
mod healthcare_circuit;

use healthcare_circuit::{artifacts_dir, build_healthcare_circuit, high_risk_witness};
use serde::Serialize;
use std::error::Error;

const MODULUS: u64 = 2013265921;

#[derive(Serialize)]
struct QuotientDump {
    modulus: u64,
    coefficients: Vec<u64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let circuit = build_healthcare_circuit(MODULUS);
    let witness = high_risk_witness(MODULUS);

    let q_coeffs = circuit
        .compute_quotient_poly(&witness)
        .expect("witness must satisfy constraints");

    let dump = QuotientDump {
        modulus: MODULUS,
        coefficients: q_coeffs,
    };

    let json = serde_json::to_string_pretty(&dump)?;
    println!("{}", json);

    let mut suggested_path = artifacts_dir();
    suggested_path.push("healthcare_quotient.json");
    println!(
        "Suggested quotient dump destination: {}",
        suggested_path.display()
    );

    Ok(())
}
