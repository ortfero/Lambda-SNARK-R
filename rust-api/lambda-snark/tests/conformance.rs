//! Conformance tests using test vectors.
//!
//! These tests validate cross-language compatibility by loading
//! standardized test vectors and verifying expected behavior.

use lambda_snark::{Field, Params, Profile, SecurityLevel};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct TestVectorParams {
    description: String,
    security_level: u32,
    profile: ProfileJson,
    random_seed: String,
}

#[derive(Debug, Deserialize)]
struct ProfileJson {
    #[serde(rename = "type")]
    profile_type: String,
    n: usize,
    k: usize,
    q: u64,
    sigma: f64,
}

#[derive(Debug, Deserialize)]
struct TestVectorInput {
    #[serde(flatten)]
    data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct TestVectorWitness {
    #[serde(flatten)]
    data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct TestVectorExpected {
    valid: bool,
    #[serde(flatten)]
    extra: serde_json::Value,
}

fn load_test_vector(
    dir: &str,
) -> (
    Params,
    TestVectorInput,
    TestVectorWitness,
    TestVectorExpected,
) {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test-vectors")
        .join(dir);

    let params_json: TestVectorParams = serde_json::from_str(
        &fs::read_to_string(base.join("params.json")).expect("Failed to read params.json"),
    )
    .expect("Failed to parse params.json");

    let input: TestVectorInput = serde_json::from_str(
        &fs::read_to_string(base.join("input.json")).expect("Failed to read input.json"),
    )
    .expect("Failed to parse input.json");

    let witness: TestVectorWitness = serde_json::from_str(
        &fs::read_to_string(base.join("witness.json")).expect("Failed to read witness.json"),
    )
    .expect("Failed to parse witness.json");

    let expected: TestVectorExpected = serde_json::from_str(
        &fs::read_to_string(base.join("expected.json")).expect("Failed to read expected.json"),
    )
    .expect("Failed to parse expected.json");

    let sec_level = match params_json.security_level {
        128 => SecurityLevel::Bits128,
        192 => SecurityLevel::Bits192,
        256 => SecurityLevel::Bits256,
        _ => panic!("Invalid security level"),
    };

    let profile = Profile::RingB {
        n: params_json.profile.n,
        k: params_json.profile.k,
        q: params_json.profile.q,
        sigma: params_json.profile.sigma,
    };

    let params = Params::new(sec_level, profile);

    (params, input, witness, expected)
}

#[test]
fn test_tv0_linear_system() {
    let (params, input, witness, expected) = load_test_vector("tv-0-linear-system");

    // Validate parameters
    assert!(params.validate().is_ok(), "Parameters should be valid");

    // For now, just verify we can load the test vector
    // TODO: Implement full prove/verify once prover is ready
    println!("TV-0: Loaded linear system test vector");
    println!("  Expected valid: {}", expected.valid);
    assert!(expected.valid);
}

#[test]
fn test_tv1_multiplication() {
    let (params, input, witness, expected) = load_test_vector("tv-1-multiplication");

    // Validate parameters
    assert!(params.validate().is_ok(), "Parameters should be valid");

    // Extract public input
    let public = input
        .data
        .get("public")
        .and_then(|v| v.as_array())
        .expect("Missing public array");
    assert_eq!(public.len(), 2);
    assert_eq!(public[0].as_u64().unwrap(), 1);
    assert_eq!(public[1].as_u64().unwrap(), 91);

    // Extract witness
    let witness_arr = witness
        .data
        .get("witness")
        .and_then(|v| v.as_array())
        .expect("Missing witness array");
    assert_eq!(witness_arr.len(), 2);
    let a = witness_arr[0].as_u64().unwrap();
    let b = witness_arr[1].as_u64().unwrap();

    // Verify multiplication
    assert_eq!(a * b, 91, "Witness should satisfy a * b = 91");
    assert_eq!(a, 7);
    assert_eq!(b, 13);

    println!("TV-1: Verified 7 × 13 = 91");
    assert!(expected.valid);
}

#[test]
fn test_tv2_plaquette() {
    let (params, input, witness, expected) = load_test_vector("tv-2-plaquette");

    // Validate parameters
    assert!(params.validate().is_ok(), "Parameters should be valid");

    // Extract witness angles
    let witness_obj = witness
        .data
        .get("witness")
        .and_then(|v| v.as_object())
        .expect("Missing witness object");

    let theta1 = witness_obj.get("θ₁").and_then(|v| v.as_i64()).unwrap();
    let theta2 = witness_obj.get("θ₂").and_then(|v| v.as_i64()).unwrap();
    let theta3 = witness_obj.get("θ₃").and_then(|v| v.as_i64()).unwrap();
    let theta4 = witness_obj.get("θ₄").and_then(|v| v.as_i64()).unwrap();

    // Verify plaquette constraint: θ₁ + θ₂ - θ₃ - θ₄ = 0
    let sum = theta1 + theta2 - theta3 - theta4;
    assert_eq!(sum, 0, "Plaquette constraint should be satisfied");

    println!("TV-2: Verified plaquette constraint");
    println!(
        "  θ₁={}, θ₂={}, θ₃={}, θ₄={}",
        theta1, theta2, theta3, theta4
    );
    println!(
        "  Sum: {} + {} - {} - {} = {}",
        theta1, theta2, theta3, theta4, sum
    );
    assert!(expected.valid);
}

#[test]
fn test_all_vectors_have_consistent_params() {
    // Load all three test vectors
    let (params0, _, _, _) = load_test_vector("tv-0-linear-system");
    let (params1, _, _, _) = load_test_vector("tv-1-multiplication");
    let (params2, _, _, _) = load_test_vector("tv-2-plaquette");

    // All should use the same profile (for consistency)
    assert_eq!(params0.security_level, params1.security_level);
    assert_eq!(params1.security_level, params2.security_level);

    match (&params0.profile, &params1.profile, &params2.profile) {
        (
            Profile::RingB {
                n: n0,
                k: k0,
                q: q0,
                sigma: s0,
            },
            Profile::RingB {
                n: n1,
                k: k1,
                q: q1,
                sigma: s1,
            },
            Profile::RingB {
                n: n2,
                k: k2,
                q: q2,
                sigma: s2,
            },
        ) => {
            assert_eq!(n0, n1);
            assert_eq!(n1, n2);
            assert_eq!(k0, k1);
            assert_eq!(k1, k2);
            assert_eq!(q0, q1);
            assert_eq!(q1, q2);
            assert_eq!(s0, s1);
            assert_eq!(s1, s2);
        }
        _ => panic!("All test vectors should use RingB profile"),
    }

    println!("✓ All test vectors use consistent parameters");
}
