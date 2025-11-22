# Contributing to ΛSNARK-R

Thank you for your interest in contributing to ΛSNARK-R! This document provides guidelines and instructions for contributors.

## 🎯 Development Philosophy

We follow **Value-Driven Development** with emphasis on:
1. **Security First**: No compromise on cryptographic correctness
2. **Formal Verification**: Lean 4 proofs for critical properties
3. **Reproducibility**: Deterministic builds, fixed seeds, extensive logging
4. **Quality over Speed**: Thorough review > rapid merges

## 🔄 Development Workflow (Trunk-Based Development)

1. **Clone and setup**:
   ```bash
   git clone https://github.com/URPKS/lambda-snark-r.git
   cd lambda-snark-r
   make setup  # Installs dependencies, pre-commit hooks
   ```

2. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-123
   ```

3. **Make changes**:
   - Follow code style guidelines (see below)
   - Add tests for new functionality
   - Update documentation

4. **Run checks locally**:
   ```bash
   make lint        # Linters (clippy, clang-tidy)
   make test        # All tests
   make test-ct     # Constant-time validation (dudect)
   ```

5. **Commit with Conventional Commits**:
   ```bash
   git commit -m "feat(prover): add rejection sampling with timeout"
   git commit -m "fix(verifier): constant-time comparison in VC::verify"
   git commit -m "docs: update API reference for Commitment"
   ```

   Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

6. **Submit Pull Request**:
   - Title: Clear, descriptive (e.g., "Add AVX-512 optimization for NTT")
   - Description: Link to issue, explain changes, benchmark results if applicable
   - Mark as draft if work-in-progress

## 📝 Code Style

### C++ (cpp-core/)

- **Standard**: C++17 (compatible with C++20)
- **Formatter**: `clang-format` (see `.clang-format`)
- **Linter**: `clang-tidy`

```cpp
// Good: const-correct, explicit types, doxygen comments
/**
 * @brief Commit to a message using LWE encryption.
 * @param ctx LWE context (contains public parameters).
 * @param message Plaintext message vector.
 * @param msg_len Length of message.
 * @param seed Random seed for reproducibility.
 * @return Pointer to LweCommitment, or nullptr on failure.
 */
extern "C" LweCommitment* lwe_commit(
    const LweContext* ctx,
    const uint64_t* message,
    size_t msg_len,
    uint64_t seed
) noexcept;

// Bad: no docs, mutable params, unchecked casts
LweCommitment* commit(LweContext* c, uint64_t* msg, int len, uint64_t s) {
    return (LweCommitment*)malloc(sizeof(LweCommitment));
}
```

**Key rules**:
- No exceptions across FFI boundary (`noexcept`)
- Const-correct parameters
- RAII for resource management
- Doxygen comments for all public APIs

### Rust (rust-api/)

- **Edition**: 2021
- **Formatter**: `rustfmt` (see `rustfmt.toml`)
- **Linter**: `clippy` with strict settings

```rust
// Good: explicit lifetimes, error handling, doc comments
/// Commit to a witness vector with ZK masking.
///
/// # Arguments
/// * `ctx` - LWE context containing public parameters
/// * `message` - Witness vector (will be zeroized on drop)
/// * `seed` - Random seed for deterministic testing
///
/// # Errors
/// Returns `Error::CommitmentFailed` if C++ core returns null.
///
/// # Safety
/// This function performs FFI calls; memory safety guaranteed by wrapper.
pub fn commit(
    ctx: &LweContext,
    message: &[Field],
    seed: u64,
) -> Result<Commitment, Error> {
    let inner = unsafe {
        ffi::lwe_commit(ctx.as_ptr(), message.as_ptr(), message.len(), seed)
    };
    
    if inner.is_null() {
        return Err(Error::CommitmentFailed);
    }
    
    Ok(Commitment { inner })
}

// Bad: unwrap, no docs, public unsafe
pub fn commit(ctx: &LweContext, msg: &[Field]) -> Commitment {
    let ptr = unsafe { ffi::lwe_commit(ctx.ptr, msg.as_ptr(), msg.len(), 0) };
    Commitment { inner: ptr.unwrap() }  // PANIC!
}
```

**Key rules**:
- `#![deny(unsafe_op_in_unsafe_fn)]`
- `#![warn(missing_docs)]`
- Never `unwrap()` in library code (use `?` or `expect` with context)
- Zeroize secrets with `#[derive(Zeroize)]`

### Lean 4 (formal/)

```lean
-- Good: explicit types, detailed proof sketch
theorem knowledge_soundness
    {F : Type} [Field F] [Fintype F] [DecidableEq F]
    (VC : VectorCommitment F) (cs : R1CS F) (secParam : ℕ)
    (A : Adversary F VC) (ε : ℕ → ℝ)
    (assumptions : SoundnessAssumptions F VC cs)
    (provider : ForkingEquationsProvider VC cs)
    (h_rom : RandomOracleModel VC) :
    ∃ (E : Extractor F VC), E.poly_time ∧
        ∀ x,
            (∃ π, verify VC cs x π = true) →
            ∃ w, satisfies cs w ∧ extractPublic cs.h_pub_le w = x := by
  -- 1. Apply forking lemma to rewind A
    have fork := forking_lemma VC cs secParam A ε assumptions provider h_rom
  -- 2. Extract collision from two accepting transcripts
  have collision := extract_collision fork
  -- 3. Collision contradicts SIS hardness
    exact collision.contradicts assumptions.moduleSIS_holds
```

## 🧪 Testing Guidelines

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_commitment_correctness() {
        let ctx = LweContext::new_for_testing();
        let msg = vec![1, 2, 3];
        let comm = commit(&ctx, &msg, 0x1234).unwrap();
        
        // Verify opening
        assert!(comm.verify_opening(&ctx, &msg, &/* randomness */));
    }
    
    #[test]
    fn test_commitment_hiding() {
        // Statistical test: distributions indistinguishable
        // ...
    }
}
```

### Property-Based Tests (proptest)

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_commitment_binding(
        msg1 in prop::collection::vec(any::<u64>(), 10),
        msg2 in prop::collection::vec(any::<u64>(), 10),
    ) {
        prop_assume!(msg1 != msg2);
        let ctx = LweContext::new_for_testing();
        let comm1 = commit(&ctx, &msg1, 0).unwrap();
        let comm2 = commit(&ctx, &msg2, 0).unwrap();
        
        // Same commitment => same message (binding)
        if comm1.as_bytes() == comm2.as_bytes() {
            prop_assert_eq!(msg1, msg2);
        }
    }
}
```

### Conformance Tests (TV-0/1/2)

See `tests/conformance/` for test vectors.

## 🔒 Security Considerations

### Constant-Time Code

**MUST**:
- Use `subtle::ConstantTimeEq` for secret comparisons
- No branches on secret data
- Validate with `dudect-bencher`

```rust
use subtle::ConstantTimeEq;

// Good
pub fn verify_tag(computed: &[u8], claimed: &[u8]) -> bool {
    computed.ct_eq(claimed).into()
}

// Bad: timing leak
pub fn verify_tag_bad(computed: &[u8], claimed: &[u8]) -> bool {
    computed == claimed  // Early return on first mismatch!
}
```

### Zeroization

```rust
use zeroize::{Zeroize, ZeroizeOnDrop};

#[derive(Zeroize, ZeroizeOnDrop)]
pub struct Secret {
    key: Vec<u8>,
}

impl Drop for Secret {
    fn drop(&mut self) {
        // Automatically zeroizes on drop
    }
}
```

### Fuzzing

```bash
# Add fuzz target
cargo fuzz add fuzz_prove

# Run fuzzer
cargo fuzz run fuzz_prove -- -max_total_time=3600
```

## 📚 Documentation

- **Inline docs**: Required for all public APIs
- **Architecture docs**: Update `docs/architecture/` for design changes
- **Changelog**: Add entry to `CHANGELOG.md` (auto-generated from commits)

## 🚫 What NOT to Contribute

- ❌ **Breaking changes without discussion** (open issue first)
- ❌ **Code without tests** (minimum 80% coverage)
- ❌ **Unsafe code without justification** (explain in PR)
- ❌ **Non-constant-time crypto** (will be rejected)

## 📞 Getting Help

- **Discord**: [URPKS Community](https://discord.gg/urpks)
- **Discussions**: [GitHub Discussions](https://github.com/URPKS/lambda-snark-r/discussions)
- **Email**: dev@lambda-snark.org

## 🏆 Recognition

Contributors will be listed in:
- `CONTRIBUTORS.md`
- GitHub insights
- Academic papers (if significant contribution)

Thank you for contributing to post-quantum cryptography! 🎉
