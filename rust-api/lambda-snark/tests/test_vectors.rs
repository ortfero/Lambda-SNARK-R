use lambda_snark_core::r1cs::{SparseMatrix, R1CS};
use serde_json::Value;
use std::fs;

const TEST_MODULUS: u64 = 17592186044417; // 2^44 + 1

fn load_test_vector(tv_name: &str) -> (Vec<u64>, R1CS) {
    // Path from workspace root
    let workspace_root = std::env::var("CARGO_MANIFEST_DIR")
        .map(|p| {
            std::path::PathBuf::from(p)
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf()
        })
        .unwrap_or_else(|_| std::path::PathBuf::from(".."));

    let path = workspace_root
        .join("test-vectors")
        .join(tv_name)
        .join("constraints.json");
    let data = fs::read_to_string(&path).unwrap_or_else(|_| panic!("Failed to read {:?}", path));

    let json: Value = serde_json::from_str(&data).unwrap();

    // Extract witness
    let witness_arr = json["verification"]["witness"].as_array().unwrap();
    let witness: Vec<u64> = witness_arr.iter().map(|v| v.as_u64().unwrap()).collect();

    // Extract m, n
    let m = json["m"].as_u64().unwrap() as u32;
    let n = json["n"].as_u64().unwrap() as u32;

    // Extract constraints
    let constraints = json["constraints"].as_array().unwrap();

    // Build matrices (assuming single constraint for simplicity)
    let constraint = &constraints[0];

    let a_entries = parse_sparse_matrix(&constraint["A"]);
    let b_entries = parse_sparse_matrix(&constraint["B"]);
    let c_entries = parse_sparse_matrix(&constraint["C"]);

    let a_matrix = SparseMatrix::from_entries(m, n, a_entries);
    let b_matrix = SparseMatrix::from_entries(m, n, b_entries);
    let c_matrix = SparseMatrix::from_entries(m, n, c_entries);

    let r1cs =
        R1CS::new(a_matrix, b_matrix, c_matrix, TEST_MODULUS).expect("Failed to create R1CS");

    (witness, r1cs)
}

fn parse_sparse_matrix(json: &Value) -> Vec<(u32, u32, u64)> {
    json.as_array()
        .map(|arr| {
            arr.iter()
                .map(|entry| {
                    let row = entry["row"].as_u64().unwrap() as u32;
                    let col = entry["col"].as_u64().unwrap() as u32;
                    let value = entry["value"].as_i64().unwrap() as u64;
                    (row, col, value)
                })
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn test_tv1_constraints() {
    let (witness, r1cs) = load_test_vector("tv-1-multiplication");

    // Validate witness structure
    assert_eq!(witness.len(), 4);
    assert_eq!(witness[0], 1); // z[0] = 1
    assert_eq!(witness[1], 7); // a = 7
    assert_eq!(witness[2], 13); // b = 13
    assert_eq!(witness[3], 91); // c = 91

    // Validate dimensions
    assert_eq!(r1cs.num_constraints(), 1);
    assert_eq!(r1cs.num_variables(), 4);

    // Validate constraint satisfaction
    let result = r1cs
        .validate_witness(&witness)
        .expect("FFI call should succeed");
    assert!(result, "Witness should satisfy constraints");
}

#[test]
fn test_tv2_constraints() {
    let (witness, r1cs) = load_test_vector("tv-2-plaquette");

    // Validate witness structure
    assert_eq!(witness.len(), 5);
    assert_eq!(witness[0], 1); // z[0] = 1
    assert_eq!(witness[1], 314); // θ₁
    assert_eq!(witness[2], 628); // θ₂
    assert_eq!(witness[3], 471); // θ₃
    assert_eq!(witness[4], 471); // θ₄

    // Validate dimensions
    assert_eq!(r1cs.num_constraints(), 1);
    assert_eq!(r1cs.num_variables(), 5);

    // Validate constraint satisfaction
    let result = r1cs
        .validate_witness(&witness)
        .expect("FFI call should succeed");
    assert!(result, "Witness should satisfy constraints");

    // Verify plaquette closure: θ₁ + θ₂ - θ₃ - θ₄ = 0
    let sum = (witness[1] + witness[2]) - (witness[3] + witness[4]);
    assert_eq!(sum, 0);
}

#[test]
fn test_tv1_invalid_witness() {
    let (mut witness, r1cs) = load_test_vector("tv-1-multiplication");

    // Corrupt witness: change factor a=7 to a=8 (but keep c=91)
    // This breaks a × b = c since 8 × 13 = 104 ≠ 91
    witness[1] = 8;

    // Should fail validation
    let result = r1cs
        .validate_witness(&witness)
        .expect("FFI call should succeed");
    assert!(
        !result,
        "Witness should NOT satisfy constraints after corruption"
    );
}

#[test]
fn test_tv2_invalid_witness() {
    let (mut witness, r1cs) = load_test_vector("tv-2-plaquette");

    // Corrupt witness: change θ₁ to break plaquette closure
    witness[1] = 315; // was 314

    // Should fail validation
    let result = r1cs
        .validate_witness(&witness)
        .expect("FFI call should succeed");
    assert!(
        !result,
        "Witness should NOT satisfy constraints after corruption"
    );
}
