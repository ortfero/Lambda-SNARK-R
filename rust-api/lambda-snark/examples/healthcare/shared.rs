//! Shared healthcare diagnosis circuit routines used across examples.
//!
//! Provides builders for the R1CS instance, witness constructors, and
//! a convenience driver that exports the circuit as a Lean artifact.

use std::fs;
use std::path::PathBuf;

use lambda_snark::{CircuitBuilder, LeanExportable, R1CS};

/// Build healthcare diagnosis circuit with binary threshold flags.
pub fn build_healthcare_circuit(modulus: u64) -> R1CS {
    let mut builder = CircuitBuilder::new(modulus);

    // Allocate variables (PUBLIC INPUTS FIRST)
    let one = builder.alloc_var(); // z_0 = 1 (PUBLIC constant)
    let risk_score = builder.alloc_var(); // z_1 = 3 (PUBLIC OUTPUT)

    // Private health metrics (not constrained directly, comparison done by prover)
    let _glucose = builder.alloc_var(); // z_2 = 142 mg/dL (PRIVATE, unconstrained)
    let _age = builder.alloc_var(); // z_3 = 45 years (PRIVATE, unconstrained)
    let _bmi = builder.alloc_var(); // z_4 = 31 kg/m² (PRIVATE, unconstrained)

    // Threshold flags (prover sets these based on comparisons)
    let glucose_high = builder.alloc_var(); // z_5: 1 if glucose > 126, else 0
    let age_high = builder.alloc_var(); // z_6: 1 if age > 40, else 0
    let bmi_high = builder.alloc_var(); // z_7: 1 if BMI > 30, else 0

    // === BINARY CONSTRAINTS ===
    builder.add_constraint(
        vec![(glucose_high, 1)],
        vec![(glucose_high, 1), (one, modulus - 1)],
        vec![],
    );
    builder.add_constraint(
        vec![(age_high, 1)],
        vec![(age_high, 1), (one, modulus - 1)],
        vec![],
    );
    builder.add_constraint(
        vec![(bmi_high, 1)],
        vec![(bmi_high, 1), (one, modulus - 1)],
        vec![],
    );

    // === AND GATE: all_high = glucose_high ∧ age_high ∧ bmi_high ===
    let temp = builder.alloc_var();
    let all_high = builder.alloc_var();

    builder.add_constraint(
        vec![(glucose_high, 1)],
        vec![(age_high, 1)],
        vec![(temp, 1)],
    );
    builder.add_constraint(vec![(temp, 1)], vec![(bmi_high, 1)], vec![(all_high, 1)]);

    // === RISK SCORE COMPUTATION ===
    builder.add_constraint(
        vec![(one, 1), (all_high, 2)],
        vec![(one, 1)],
        vec![(risk_score, 1)],
    );

    // === PADDING CONSTRAINTS (7-10) ===
    builder.add_constraint(vec![], vec![], vec![]);
    builder.add_constraint(vec![], vec![], vec![]);
    builder.add_constraint(vec![], vec![], vec![]);
    builder.add_constraint(vec![], vec![], vec![]);

    builder.set_public_inputs(2);

    let circuit = builder.build();

    debug_assert!(
        circuit.is_satisfied(&high_risk_witness(modulus)),
        "High-risk witness must satisfy healthcare circuit",
    );
    debug_assert!(
        circuit.is_satisfied(&low_risk_witness(modulus)),
        "Low-risk witness must satisfy healthcare circuit",
    );

    circuit
}

/// Prepare witness for HIGH risk patient.
pub fn high_risk_witness(_modulus: u64) -> Vec<u64> {
    let glucose = 142;
    let age = 45;
    let bmi = 31;

    let glucose_high = if glucose > 126 { 1 } else { 0 };
    let age_high = if age > 40 { 1 } else { 0 };
    let bmi_high = if bmi > 30 { 1 } else { 0 };

    let temp = glucose_high * age_high;
    let all_high = temp * bmi_high;
    let risk_score = 1 + 2 * all_high;

    vec![
        1,
        risk_score,
        glucose,
        age,
        bmi,
        glucose_high,
        age_high,
        bmi_high,
        temp,
        all_high,
    ]
}

/// Prepare witness for LOW risk patient.
pub fn low_risk_witness(_modulus: u64) -> Vec<u64> {
    let glucose = 95;
    let age = 28;
    let bmi = 23;

    let glucose_high = 0;
    let age_high = 0;
    let bmi_high = 0;
    let temp = 0;
    let all_high = 0;
    let risk_score = 1;

    vec![
        1,
        risk_score,
        glucose,
        age,
        bmi,
        glucose_high,
        age_high,
        bmi_high,
        temp,
        all_high,
    ]
}

/// Location for Lean artifacts emitted by the example.
pub fn artifacts_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("artifacts")
        .join("r1cs")
}

/// Run the healthcare circuit example end-to-end.
#[allow(dead_code)]
pub fn run_example() -> Result<(), Box<dyn std::error::Error>> {
    let modulus = 2013265921u64;
    let circuit = build_healthcare_circuit(modulus);
    let witness = high_risk_witness(modulus);
    let low_risk = low_risk_witness(modulus);

    if !circuit.is_satisfied(&witness) {
        return Err("HIGH risk witness must satisfy the healthcare circuit".into());
    }

    if !circuit.is_satisfied(&low_risk) {
        return Err("LOW risk witness must satisfy the healthcare circuit".into());
    }

    let public_inputs = circuit.public_inputs(&witness);
    if public_inputs.len() != 2 || public_inputs[1] != 3 {
        return Err("Unexpected public inputs for healthcare circuit".into());
    }

    println!(
        "Healthcare circuit satisfied. Risk score = {} (expected 3)",
        public_inputs[1]
    );

    let lean_term = circuit.to_lean_term();
    let mut artifact_path = artifacts_dir();
    fs::create_dir_all(&artifact_path)?;
    artifact_path.push("healthcare.term");
    fs::write(&artifact_path, lean_term.as_bytes())?;

    println!(
        "Lean verification key written to {}",
        artifact_path.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_risk_circuit() {
        let modulus = 2013265921u64;
        let circuit = build_healthcare_circuit(modulus);
        let witness = high_risk_witness(modulus);

        assert!(
            circuit.is_satisfied(&witness),
            "HIGH risk witness must satisfy R1CS",
        );

        let public_inputs = circuit.public_inputs(&witness);
        assert_eq!(public_inputs.len(), 2);
        assert_eq!(public_inputs[1], 3);
    }

    #[test]
    fn test_low_risk_circuit() {
        let modulus = 2013265921u64;
        let circuit = build_healthcare_circuit(modulus);
        let witness = low_risk_witness(modulus);

        assert!(
            circuit.is_satisfied(&witness),
            "LOW risk witness must satisfy R1CS",
        );

        let public_inputs = circuit.public_inputs(&witness);
        assert_eq!(public_inputs[1], 1);
    }

    #[test]
    fn test_run_example_smoke() {
        run_example().expect("Healthcare example should run end-to-end");
        let path = artifacts_dir();
        assert!(path.ends_with("artifacts/r1cs"));
    }
}
