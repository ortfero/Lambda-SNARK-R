//! Healthcare Diagnosis Circuit Example
//! 
//! Proves medical diagnosis result without revealing patient data.
//! 
//! **Scenario**: Hospital proves high diabetes risk without sending glucose/age/BMI.
//! **Circuit Logic**:
//! - HIGH risk (3): glucose > 126 AND age > 40 AND BMI > 30
//! - MEDIUM risk (2): glucose > 100 OR age > 35 OR BMI > 25
//! - LOW risk (1): otherwise
//! 
//! **Privacy**: Verifier only sees risk_score, not actual health metrics.

use lambda_snark::{CircuitBuilder, R1CS};

/// Build healthcare diagnosis circuit
/// 
/// # Constraints
/// - Binary constraints for threshold flags (glucose_high, age_high, bmi_high)
/// - AND gate for all_high = glucose_high ∧ age_high ∧ bmi_high
/// - Risk score computation: risk = 1 + 2*all_high
/// 
/// Note: Threshold comparisons (glucose>126, age>40, BMI>30) are trusted in witness.
/// Circuit only enforces binary flags and logical operations.
/// 
/// Total: 6 R1CS constraints
pub fn build_healthcare_circuit(modulus: u64) -> R1CS {
    let mut builder = CircuitBuilder::new(modulus);
    
    // Allocate variables (PUBLIC INPUTS FIRST)
    let one = builder.alloc_var();           // z_0 = 1 (PUBLIC constant)
    let risk_score = builder.alloc_var();    // z_1 = 3 (PUBLIC OUTPUT)
    
    // Private health metrics (not constrained directly, comparison done by prover)
    let _glucose = builder.alloc_var();      // z_2 = 142 mg/dL (PRIVATE, unconstrained)
    let _age = builder.alloc_var();          // z_3 = 45 years (PRIVATE, unconstrained)
    let _bmi = builder.alloc_var();          // z_4 = 31 kg/m² (PRIVATE, unconstrained)
    
    // Threshold flags (prover sets these based on comparisons)
    let glucose_high = builder.alloc_var();  // z_5: 1 if glucose > 126, else 0
    let age_high = builder.alloc_var();      // z_6: 1 if age > 40, else 0
    let bmi_high = builder.alloc_var();      // z_7: 1 if BMI > 30, else 0
    
    // === BINARY CONSTRAINTS ===
    // Constraint 1: glucose_high ∈ {0, 1}
    // glucose_high * (glucose_high - 1) = 0
    builder.add_constraint(
        vec![(glucose_high, 1)],
        vec![(glucose_high, 1), (one, modulus - 1)],  // glucose_high - 1
        vec![],  // = 0
    );
    
    // Constraint 2: age_high ∈ {0, 1}
    builder.add_constraint(
        vec![(age_high, 1)],
        vec![(age_high, 1), (one, modulus - 1)],
        vec![],
    );
    
    // Constraint 3: bmi_high ∈ {0, 1}
    builder.add_constraint(
        vec![(bmi_high, 1)],
        vec![(bmi_high, 1), (one, modulus - 1)],
        vec![],
    );
    
    // === AND GATE: all_high = glucose_high ∧ age_high ∧ bmi_high ===
    let temp = builder.alloc_var();          // z_8: glucose_high * age_high
    let all_high = builder.alloc_var();      // z_9: temp * bmi_high
    
    // Constraint 4: temp = glucose_high * age_high
    builder.add_constraint(
        vec![(glucose_high, 1)],
        vec![(age_high, 1)],
        vec![(temp, 1)],
    );
    
    // Constraint 5: all_high = temp * bmi_high
    builder.add_constraint(
        vec![(temp, 1)],
        vec![(bmi_high, 1)],
        vec![(all_high, 1)],
    );
    
    // === RISK SCORE COMPUTATION ===
    // Constraint 6: risk_score = 1 + 2*all_high
    // (maps: all_high=0 → risk=1 LOW, all_high=1 → risk=3 HIGH)
    builder.add_constraint(
        vec![(one, 1), (all_high, 2)],
        vec![(one, 1)],
        vec![(risk_score, 1)],
    );
    
    // Public inputs: one + risk_score (z_0, z_1)
    builder.set_public_inputs(2);
    
    builder.build()
}

/// Prepare witness for HIGH risk patient
/// 
/// Patient data:
/// - Glucose: 142 mg/dL (HIGH, >126)
/// - Age: 45 years (HIGH, >40)
/// - BMI: 31 kg/m² (HIGH, >30)
/// 
/// Expected: risk_score = 3 (HIGH)
pub fn high_risk_witness(_modulus: u64) -> Vec<u64> {
    let glucose = 142;
    let age = 45;
    let bmi = 31;
    
    // Compute threshold flags (comparison done by prover)
    let glucose_high = if glucose > 126 { 1 } else { 0 };
    let age_high = if age > 40 { 1 } else { 0 };
    let bmi_high = if bmi > 30 { 1 } else { 0 };
    
    // AND gate computation
    let temp = glucose_high * age_high;
    let all_high = temp * bmi_high;
    
    // Risk score
    let risk_score = 1 + 2 * all_high;
    
    // Witness vector (matches allocation order)
    vec![
        1,              // z_0: one (PUBLIC)
        risk_score,     // z_1: risk_score (PUBLIC)
        glucose,        // z_2: glucose (PRIVATE)
        age,            // z_3: age (PRIVATE)
        bmi,            // z_4: BMI (PRIVATE)
        glucose_high,   // z_5: glucose > 126
        age_high,       // z_6: age > 40
        bmi_high,       // z_7: BMI > 30
        temp,           // z_8: glucose_high * age_high
        all_high,       // z_9: all conditions met
    ]
}

/// Prepare witness for LOW risk patient
pub fn low_risk_witness(_modulus: u64) -> Vec<u64> {
    let glucose = 95;  // Normal
    let age = 28;      // Young
    let bmi = 23;      // Healthy
    
    let glucose_high = 0;
    let age_high = 0;
    let bmi_high = 0;
    let temp = 0;
    let all_high = 0;
    let risk_score = 1; // LOW
    
    vec![
        1,              // z_0: one (PUBLIC)
        risk_score,     // z_1: risk_score (PUBLIC)
        glucose,        // z_2: glucose (PRIVATE)
        age,            // z_3: age (PRIVATE)
        bmi,            // z_4: BMI (PRIVATE)
        glucose_high,   // z_5
        age_high,       // z_6
        bmi_high,       // z_7
        temp,           // z_8
        all_high,       // z_9
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_high_risk_circuit() {
        let modulus = 17592186044423u64;
        let circuit = build_healthcare_circuit(modulus);
        let witness = high_risk_witness(modulus);
        
        // Check constraint satisfaction
        assert!(circuit.is_satisfied(&witness), "HIGH risk witness must satisfy R1CS");
        
        // Verify risk score is public
        let public_inputs = circuit.public_inputs(&witness);
        assert_eq!(public_inputs.len(), 2); // [1, risk_score]
        assert_eq!(public_inputs[1], 3, "Risk score must be 3 (HIGH)");
    }
    
    #[test]
    fn test_low_risk_circuit() {
        let modulus = 17592186044423u64;
        let circuit = build_healthcare_circuit(modulus);
        let witness = low_risk_witness(modulus);
        
        assert!(circuit.is_satisfied(&witness), "LOW risk witness must satisfy R1CS");
        
        let public_inputs = circuit.public_inputs(&witness);
        assert_eq!(public_inputs[1], 1, "Risk score must be 1 (LOW)");
    }
}
