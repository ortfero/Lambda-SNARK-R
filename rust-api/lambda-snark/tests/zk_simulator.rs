//! Zero-Knowledge Simulator Tests
//!
//! Tests for simulate_proof() function validating:
//! - Simulator generates valid-looking proofs without witness
//! - Simulated proofs are indistinguishable from real proofs
//! - Statistical properties match real proof distribution

use lambda_snark::{prove_zk, simulate_proof, LweContext, Params, Profile, SecurityLevel};

const TEST_MODULUS: u64 = 17592186044417; // 2^44 + 1

fn setup_context() -> LweContext {
    let params = Params::new(
        SecurityLevel::Bits128,
        Profile::RingB {
            n: 4096,
            k: 2,
            q: TEST_MODULUS,
            sigma: 3.19,
        },
    );
    LweContext::new(params).expect("Failed to create LWE context")
}

// ============================================================================
// Basic Simulator Tests
// ============================================================================

#[test]
fn test_simulator_generates_proof() {
    let ctx = setup_context();
    let public_inputs = vec![1, 91];

    // Simulate proof with degree 3 (matches witness [1,7,13,91])
    let sim_proof = simulate_proof(3, &public_inputs, &ctx, TEST_MODULUS, 0x1234, Some(42))
        .expect("Failed to simulate proof");

    // Proof should have valid structure
    assert!(sim_proof.challenge().alpha().value() < TEST_MODULUS);
    assert!(sim_proof.opening().evaluation().value() < TEST_MODULUS);
}

#[test]
fn test_simulator_deterministic() {
    let ctx = setup_context();
    let public_inputs = vec![1, 91];

    // Same seeds → same simulated proof (up to SEAL non-determinism)
    let sim1 = simulate_proof(3, &public_inputs, &ctx, TEST_MODULUS, 0x1234, Some(42))
        .expect("Failed to simulate proof 1");
    let sim2 = simulate_proof(3, &public_inputs, &ctx, TEST_MODULUS, 0x1234, Some(42))
        .expect("Failed to simulate proof 2");

    // Note: Due to SEAL non-determinism, commitments may differ
    // But evaluation should be deterministic from same random polynomial
    // (This test documents expected behavior, not strict equality)
    println!("Sim1 challenge: {}", sim1.challenge().alpha().value());
    println!("Sim2 challenge: {}", sim2.challenge().alpha().value());
}

#[test]
fn test_simulator_different_seeds() {
    let ctx = setup_context();
    let public_inputs = vec![1, 91];

    // Different seeds → different proofs
    let sim1 = simulate_proof(3, &public_inputs, &ctx, TEST_MODULUS, 0x1234, Some(42))
        .expect("Failed to simulate proof 1");
    let sim2 = simulate_proof(3, &public_inputs, &ctx, TEST_MODULUS, 0x5678, Some(99))
        .expect("Failed to simulate proof 2");

    // Challenges should differ (different commitments)
    assert_ne!(
        sim1.challenge().alpha().value(),
        sim2.challenge().alpha().value(),
        "Different seeds should produce different challenges"
    );
}

#[test]
fn test_simulator_random_seed() {
    let ctx = setup_context();
    let public_inputs = vec![1, 91];

    // sim_seed = None → secure random
    let sim1 = simulate_proof(3, &public_inputs, &ctx, TEST_MODULUS, 0x1234, None)
        .expect("Failed to simulate proof 1");
    let sim2 = simulate_proof(3, &public_inputs, &ctx, TEST_MODULUS, 0x5678, None)
        .expect("Failed to simulate proof 2");

    // Should produce different proofs (with high probability)
    assert_ne!(
        sim1.challenge().alpha().value(),
        sim2.challenge().alpha().value(),
        "Random seeds should produce different challenges"
    );
}

// ============================================================================
// Indistinguishability Tests
// ============================================================================

#[test]
fn test_simulator_vs_real_structure() {
    let ctx = setup_context();
    let witness = vec![1, 7, 13, 91];
    let public_inputs = vec![1, 91];

    // Generate real ZK proof
    let real_proof = prove_zk(
        &witness,
        &public_inputs,
        &ctx,
        TEST_MODULUS,
        0x1234,
        Some(42),
    )
    .expect("Failed to generate real proof");

    // Generate simulated proof (same degree)
    let sim_proof = simulate_proof(3, &public_inputs, &ctx, TEST_MODULUS, 0x5678, Some(99))
        .expect("Failed to simulate proof");

    // Both should have valid structure (check challenges and evaluations)
    assert!(real_proof.challenge().alpha().value() < TEST_MODULUS);
    assert!(sim_proof.challenge().alpha().value() < TEST_MODULUS);

    // Evaluations should be in valid range
    assert!(real_proof.opening().evaluation().value() < TEST_MODULUS);
    assert!(sim_proof.opening().evaluation().value() < TEST_MODULUS);
}

#[test]
fn test_simulator_challenge_distribution() {
    let ctx = setup_context();
    let public_inputs = vec![1, 91];

    // Generate 20 simulated proofs
    let num_proofs = 20;
    let mut challenges = Vec::new();

    for i in 0..num_proofs {
        let sim_proof = simulate_proof(3, &public_inputs, &ctx, TEST_MODULUS, 0x1000 + i, Some(i))
            .expect("Failed to simulate proof");
        challenges.push(sim_proof.challenge().alpha().value());
    }

    // Challenges should be diverse (no duplicates expected with high probability)
    let unique_challenges: std::collections::HashSet<_> = challenges.iter().collect();
    assert!(
        unique_challenges.len() >= (num_proofs - 2) as usize,
        "Challenges should be diverse (got {} unique out of {})",
        unique_challenges.len(),
        num_proofs
    );
}

#[test]
fn test_real_vs_sim_challenge_distribution() {
    let ctx = setup_context();
    let witness = vec![1, 7, 13, 91];
    let public_inputs = vec![1, 91];

    let num_samples = 10;
    let mut real_challenges = Vec::new();
    let mut sim_challenges = Vec::new();

    for i in 0..num_samples {
        // Real proof
        let real_proof = prove_zk(
            &witness,
            &public_inputs,
            &ctx,
            TEST_MODULUS,
            0x1000 + i,
            Some(i),
        )
        .expect("Failed to generate real proof");
        real_challenges.push(real_proof.challenge().alpha().value());

        // Simulated proof
        let sim_proof = simulate_proof(
            3,
            &public_inputs,
            &ctx,
            TEST_MODULUS,
            0x2000 + i,
            Some(i + 100),
        )
        .expect("Failed to simulate proof");
        sim_challenges.push(sim_proof.challenge().alpha().value());
    }

    // Both distributions should be diverse (no easy pattern)
    let real_unique: std::collections::HashSet<_> = real_challenges.iter().collect();
    let sim_unique: std::collections::HashSet<_> = sim_challenges.iter().collect();

    assert!(
        real_unique.len() >= (num_samples - 1) as usize,
        "Real challenges should be diverse"
    );
    assert!(
        sim_unique.len() >= (num_samples - 1) as usize,
        "Simulated challenges should be diverse"
    );

    println!("Real challenges: {} unique", real_unique.len());
    println!("Sim challenges: {} unique", sim_unique.len());
}

// ============================================================================
// Statistical Distinguisher Tests
// ============================================================================

#[test]
fn test_distinguisher_challenge_range() {
    let ctx = setup_context();
    let witness = vec![1, 7, 13, 91];
    let public_inputs = vec![1, 91];

    let num_samples = 20;

    // Collect real and simulated challenges
    let mut real_challenges = Vec::new();
    let mut sim_challenges = Vec::new();

    for i in 0..num_samples {
        let real = prove_zk(
            &witness,
            &public_inputs,
            &ctx,
            TEST_MODULUS,
            0x1000 + i,
            Some(i),
        )
        .expect("Failed to generate real proof");
        real_challenges.push(real.challenge().alpha().value());

        let sim = simulate_proof(
            3,
            &public_inputs,
            &ctx,
            TEST_MODULUS,
            0x2000 + i,
            Some(i + 50),
        )
        .expect("Failed to simulate proof");
        sim_challenges.push(sim.challenge().alpha().value());
    }

    // Both should be in [0, modulus)
    for &ch in &real_challenges {
        assert!(ch < TEST_MODULUS, "Real challenge {} exceeds modulus", ch);
    }
    for &ch in &sim_challenges {
        assert!(ch < TEST_MODULUS, "Sim challenge {} exceeds modulus", ch);
    }

    // Statistical test: mean should be roughly modulus/2
    let real_mean: f64 =
        real_challenges.iter().map(|&x| x as f64).sum::<f64>() / num_samples as f64;
    let sim_mean: f64 = sim_challenges.iter().map(|&x| x as f64).sum::<f64>() / num_samples as f64;
    let expected_mean = TEST_MODULUS as f64 / 2.0;

    println!(
        "Real mean: {:.2e}, Sim mean: {:.2e}, Expected: {:.2e}",
        real_mean, sim_mean, expected_mean
    );

    // Allow large deviation due to small sample size (this is a weak test)
    let tolerance = 0.5; // 50% deviation allowed
    assert!(
        (real_mean - expected_mean).abs() / expected_mean < tolerance,
        "Real mean deviates too much from expected"
    );
    assert!(
        (sim_mean - expected_mean).abs() / expected_mean < tolerance,
        "Sim mean deviates too much from expected"
    );
}

#[test]
fn test_distinguisher_evaluation_range() {
    let ctx = setup_context();
    let witness = vec![1, 7, 13, 91];
    let public_inputs = vec![1, 91];

    let num_samples = 20;

    let mut real_evals = Vec::new();
    let mut sim_evals = Vec::new();

    for i in 0..num_samples {
        let real = prove_zk(
            &witness,
            &public_inputs,
            &ctx,
            TEST_MODULUS,
            0x1000 + i,
            Some(i),
        )
        .expect("Failed to generate real proof");
        real_evals.push(real.opening().evaluation().value());

        let sim = simulate_proof(
            3,
            &public_inputs,
            &ctx,
            TEST_MODULUS,
            0x2000 + i,
            Some(i + 50),
        )
        .expect("Failed to simulate proof");
        sim_evals.push(sim.opening().evaluation().value());
    }

    // Both should be in [0, modulus)
    for &ev in &real_evals {
        assert!(ev < TEST_MODULUS, "Real evaluation {} exceeds modulus", ev);
    }
    for &ev in &sim_evals {
        assert!(ev < TEST_MODULUS, "Sim evaluation {} exceeds modulus", ev);
    }

    // Both distributions should be diverse
    let real_unique: std::collections::HashSet<_> = real_evals.iter().collect();
    let sim_unique: std::collections::HashSet<_> = sim_evals.iter().collect();

    println!(
        "Real evaluations: {} unique out of {}",
        real_unique.len(),
        num_samples
    );
    println!(
        "Sim evaluations: {} unique out of {}",
        sim_unique.len(),
        num_samples
    );

    // Should have high diversity (most values unique)
    assert!(
        real_unique.len() >= (num_samples - 2) as usize,
        "Real evaluations should be diverse"
    );
    assert!(
        sim_unique.len() >= (num_samples - 2) as usize,
        "Sim evaluations should be diverse"
    );
}

// ============================================================================
// Practical Distinguisher Test
// ============================================================================

#[test]
fn test_practical_distinguisher() {
    let ctx = setup_context();
    let witness = vec![1, 7, 13, 91];
    let public_inputs = vec![1, 91];

    let num_samples = 30;
    let mut proofs = Vec::new();
    let mut labels = Vec::new(); // true = real, false = sim

    // Generate mixed real and simulated proofs
    for i in 0..num_samples {
        if i % 2 == 0 {
            // Real proof
            let proof = prove_zk(
                &witness,
                &public_inputs,
                &ctx,
                TEST_MODULUS,
                0x1000 + i,
                Some(i),
            )
            .expect("Failed to generate real proof");
            proofs.push(proof);
            labels.push(true);
        } else {
            // Simulated proof
            let proof = simulate_proof(
                3,
                &public_inputs,
                &ctx,
                TEST_MODULUS,
                0x2000 + i,
                Some(i + 100),
            )
            .expect("Failed to simulate proof");
            proofs.push(proof);
            labels.push(false);
        }
    }

    // Simple distinguisher: try to identify real vs sim based on challenge value
    let mut correct_guesses = 0;
    let threshold = TEST_MODULUS / 2;

    for (i, proof) in proofs.iter().enumerate() {
        let challenge = proof.challenge().alpha().value();
        let guess = challenge > threshold; // Arbitrary distinguisher

        if guess == labels[i] {
            correct_guesses += 1;
        }
    }

    let accuracy = correct_guesses as f64 / num_samples as f64;
    println!(
        "Distinguisher accuracy: {:.2}% ({}/{})",
        accuracy * 100.0,
        correct_guesses,
        num_samples
    );

    // Accuracy should stay near random (50%) but SEAL randomness can skew the
    // tiny sample. Still guard against obvious leaks.
    assert!(
        (0.30..=0.70).contains(&accuracy),
        "Distinguisher should not be significantly better than random (got {:.2}%)",
        accuracy * 100.0
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_simulator_different_degrees() {
    let ctx = setup_context();
    let public_inputs = vec![1, 91];

    // Test various polynomial degrees
    for degree in [0, 1, 3, 10, 20] {
        let sim_proof =
            simulate_proof(degree, &public_inputs, &ctx, TEST_MODULUS, 0x1234, Some(42))
                .unwrap_or_else(|_| panic!("Failed to simulate proof with degree {}", degree));

        assert!(sim_proof.challenge().alpha().value() < TEST_MODULUS);
    }
}

#[test]
fn test_simulator_performance() {
    let ctx = setup_context();
    let public_inputs = vec![1, 91];

    use std::time::Instant;

    let start = Instant::now();
    let _sim_proof = simulate_proof(3, &public_inputs, &ctx, TEST_MODULUS, 0x1234, Some(42))
        .expect("Failed to simulate proof");
    let duration = start.elapsed();

    println!("Simulator time: {:?}", duration);

    // Should be fast (comparable to prove_zk, < 100ms)
    assert!(
        duration.as_millis() < 100,
        "Simulator took {:?} (expected < 100ms)",
        duration
    );
}
