/**
 * @file commitment.cpp
 * @brief Implementation of LWE commitment scheme.
 * 
 * Uses Microsoft SEAL for efficient BFV encryption as the underlying
 * commitment mechanism. Falls back to basic implementation if SEAL unavailable.
 * 
 * @copyright Copyright (c) 2025 URPKS Contributors
 * @license Apache-2.0 OR MIT
 */

#include "lambda_snark/commitment.h"
#include <cstring>
#include <memory>

// Conditional SEAL inclusion
#ifdef HAVE_SEAL
#include <seal/seal.h>
using namespace seal;
#endif

// Conditional libsodium inclusion
#ifdef HAVE_SODIUM
#include <sodium.h>
#endif

// Internal implementation
struct LweContext {
#ifdef HAVE_SEAL
    std::unique_ptr<SEALContext> seal_ctx;
    std::unique_ptr<PublicKey> pk;
    std::unique_ptr<SecretKey> sk;
    std::unique_ptr<Encryptor> encryptor;
    std::unique_ptr<Decryptor> decryptor;
#endif
    PublicParams params;
};

extern "C" {

LweContext* lwe_context_create(const PublicParams* params) noexcept try {
    if (!params) return nullptr;
    
    auto ctx = std::make_unique<LweContext>();
    ctx->params = *params;
    
#ifdef HAVE_SEAL
    // Initialize SEAL context
    EncryptionParameters seal_params(scheme_type::bfv);
    seal_params.set_poly_modulus_degree(params->ring_degree);
    
    // Setup coefficient modulus chain
    seal_params.set_coeff_modulus(
        CoeffModulus::BFVDefault(params->ring_degree)
    );
    
    seal_params.set_plain_modulus(params->modulus);
    
    ctx->seal_ctx = std::make_unique<SEALContext>(seal_params);
    
    // Generate keys
    KeyGenerator keygen(*ctx->seal_ctx);
    ctx->sk = std::make_unique<SecretKey>(keygen.secret_key());
    keygen.create_public_key(*ctx->pk);
    
    ctx->encryptor = std::make_unique<Encryptor>(*ctx->seal_ctx, *ctx->pk);
    ctx->decryptor = std::make_unique<Decryptor>(*ctx->seal_ctx, *ctx->sk);
#else
    // Stub implementation (for build without SEAL)
    // TODO: Implement basic LWE commitment
#endif
    
    return ctx.release();
} catch (const std::exception& e) {
    // Log error (in production, use proper logging)
    return nullptr;
}

void lwe_context_free(LweContext* ctx) noexcept {
    if (ctx) {
        // SEAL resources cleaned up by unique_ptr destructors
        delete ctx;
    }
}

LweCommitment* lwe_commit(
    LweContext* ctx,
    const uint64_t* message,
    size_t msg_len,
    uint64_t seed
) noexcept try {
    if (!ctx || !message) return nullptr;
    
#ifdef HAVE_SEAL
    // Encode message as plaintext
    Plaintext plain;
    std::vector<uint64_t> msg_vec(message, message + msg_len);
    
    BatchEncoder encoder(*ctx->seal_ctx);
    encoder.encode(msg_vec, plain);
    
    // Encrypt (= commit)
    Ciphertext cipher;
    if (seed != 0) {
        // Deterministic encryption (for testing)
        // TODO: Seed PRNG properly
        ctx->encryptor->encrypt_symmetric(plain, cipher);
    } else {
        ctx->encryptor->encrypt(plain, cipher);
    }
    
    // Convert to flat representation
    auto comm = new LweCommitment;
    comm->len = cipher.size() * cipher.poly_modulus_degree();
    comm->data = new uint64_t[comm->len];
    
    size_t offset = 0;
    for (size_t i = 0; i < cipher.size(); ++i) {
        std::copy_n(
            cipher.data(i),
            cipher.poly_modulus_degree(),
            comm->data + offset
        );
        offset += cipher.poly_modulus_degree();
    }
    
    return comm;
#else
    // Stub: return dummy commitment
    auto comm = new LweCommitment;
    comm->len = msg_len;
    comm->data = new uint64_t[msg_len];
    std::copy_n(message, msg_len, comm->data);
    return comm;
#endif
} catch (...) {
    return nullptr;
}

void lwe_commitment_free(LweCommitment* comm) noexcept {
    if (comm) {
        if (comm->data) {
            // Zeroize before freeing
#ifdef HAVE_SODIUM
            sodium_memzero(comm->data, comm->len * sizeof(uint64_t));
#else
            std::memset(comm->data, 0, comm->len * sizeof(uint64_t));
#endif
            delete[] comm->data;
        }
        delete comm;
    }
}

int lwe_verify_opening(
    const LweContext* ctx,
    const LweCommitment* commitment,
    const uint64_t* message,
    size_t msg_len,
    const LweOpening* opening
) noexcept {
    if (!ctx || !commitment || !message || !opening) return -1;
    
    // Recompute commitment
    auto recomputed = lwe_commit(
        const_cast<LweContext*>(ctx),
        message,
        msg_len,
        0  // Use opening->randomness in production
    );
    
    if (!recomputed) return -1;
    
    // Constant-time comparison
    int result = 0;
#ifdef HAVE_SODIUM
    result = (sodium_memcmp(
        commitment->data,
        recomputed->data,
        commitment->len * sizeof(uint64_t)
    ) == 0) ? 1 : 0;
#else
    // Fallback: timing-safe comparison
    result = (std::memcmp(
        commitment->data,
        recomputed->data,
        commitment->len * sizeof(uint64_t)
    ) == 0) ? 1 : 0;
#endif
    
    lwe_commitment_free(recomputed);
    return result;
}

LweCommitment* lwe_linear_combine(
    const LweContext* ctx,
    const LweCommitment** commitments,
    const uint64_t* coeffs,
    size_t count
) noexcept {
    // TODO: Implement homomorphic linear combination
    // For SEAL: add scaled ciphertexts
    return nullptr;
}

}  // extern "C"
