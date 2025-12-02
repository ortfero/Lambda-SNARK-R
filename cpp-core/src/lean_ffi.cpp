/**
 * @file lean_ffi.cpp
 * @brief Lean 4 FFI for exporting VK to formal verification layer.
 * 
 * Exports R1CS verification key and security parameters to Lean term format
 * for bidirectional integration with formal proofs (M10.2).
 * 
 * @copyright Copyright (c) 2025 URPKS Contributors
 * @license Apache-2.0 OR MIT
 */

#include "lambda_snark/commitment.h"
#include "lambda_snark/r1cs.h"
#include "lambda_snark/types.h"
#include <sstream>
#include <iomanip>
#include <cstring>

#include <seal/seal.h>
using namespace seal;

#ifndef HAVE_SEAL
#error "Lambda-SNARK requires Microsoft SEAL; configure the build with SEAL support."
#endif

// Forward declare LweContext internals (defined in commitment.cpp)
// This allows access to SEAL context without full definition
struct LweContext {
    std::shared_ptr<SEALContext> seal_ctx;
    std::unique_ptr<PublicKey> pk;
    std::unique_ptr<SecretKey> sk;
    std::unique_ptr<Encryptor> encryptor;
    std::unique_ptr<Decryptor> decryptor;
    std::unique_ptr<BatchEncoder> encoder;
    std::unique_ptr<Evaluator> evaluator;
    PublicParams params;
};

namespace {

/**
 * @brief Convert sparse matrix to Lean term format.
 * 
 * Format: SparseMatrix.mk rows cols [(r0,c0,v0), (r1,c1,v1), ...]
 */
std::string sparse_matrix_to_lean(const SparseMatrix& matrix) {
    std::ostringstream oss;
    oss << "SparseMatrix.mk " << matrix.n_rows << " " << matrix.n_cols << " [";
    
    for (size_t i = 0; i < matrix.n_entries; ++i) {
        if (i > 0) oss << ", ";
        oss << "(" 
            << matrix.entries[i].row << ", "
            << matrix.entries[i].col << ", "
            << matrix.entries[i].value << ")";
    }
    
    oss << "]";
    return oss.str();
}

/**
 * @brief Export security parameters to Lean record format.
 * 
 * Format: { n := 4096, k := 2, q := 12289, σ := 3.2, λ := 128 }
 */
std::string public_params_to_lean(const PublicParams* params) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "{ n := " << params->ring_degree
        << ", k := " << params->module_rank
        << ", q := " << params->modulus
        << ", σ := " << params->sigma
        << ", λ := " << params->security_level
        << " }";
    return oss.str();
}

/**
 * @brief Export SEAL EncryptionParameters to Lean format.
 * 
 * Extracts key security parameters from SEAL context.
 * 
 * @note SEAL uses coefficient modulus chain (multiple primes).
 *       We extract only the first prime for simplicity.
 */
std::string seal_params_to_lean(const EncryptionParameters& params) {
    std::ostringstream oss;
    oss << "{ scheme := BFV"
        << ", n := " << params.poly_modulus_degree();
    
    // Extract first coefficient modulus (primary security parameter)
    auto coeff_mod = params.coeff_modulus();
    if (!coeff_mod.empty()) {
        oss << ", q := " << coeff_mod[0].value();
    }
    
    // Extract plaintext modulus
    oss << ", t := " << params.plain_modulus().value();
    
    oss << " }";
    return oss.str();
}

/**
 * @brief Serialize SEAL PublicKey to base64 string.
 * 
 * Enables export of commitment key for Lean verification.
 * 
 * @return Base64-encoded public key or empty string on error.
 */
std::string seal_pubkey_to_base64(const PublicKey& pk, const SEALContext& ctx) {
    std::ostringstream oss;
    try {
        // Serialize public key to stream
        pk.save(oss, compr_mode_type::zstd);
        
        // Convert to base64 (simplified, production needs proper base64 encoding)
        // For prototype: return hex-encoded binary
        std::string binary = oss.str();
        std::ostringstream hex_oss;
        hex_oss << std::hex << std::setfill('0');
        for (char c : binary) {
            hex_oss << std::setw(2) << static_cast<int>(c);
        }
        return hex_oss.str();
    } catch (const std::exception& e) {
        fprintf(stderr, "seal_pubkey_to_base64 error: %s\n", e.what());
        return "";
    }
}

} // anonymous namespace

extern "C" {

/**
 * @brief Export R1CS verification key to Lean term format.
 * 
 * Format: ⟨nCons, nVars, nPublic, q, A, B, C⟩
 * 
 * This enables bidirectional integration with Lean 4 formal proofs:
 * - Rust implementation → Lean verification
 * - Lean parameters → Rust validation
 * 
 * @param r1cs R1CS constraint system
 * @param params Security parameters
 * @param out_buffer Output buffer (allocated by caller)
 * @param buffer_size Size of output buffer
 * @return Number of bytes written, or -1 on error
 */
int export_vk_to_lean(
    const R1CSConstraintSystem* r1cs,
    const PublicParams* params,
    char* out_buffer,
    size_t buffer_size
) noexcept {
    if (!r1cs || !params || !out_buffer) return -1;
    
    try {
        if (r1cs->n_public_inputs > r1cs->n_vars) {
            fprintf(stderr,
                    "export_vk_to_lean: n_public_inputs (%u) exceeds n_vars (%u)\n",
                    r1cs->n_public_inputs,
                    r1cs->n_vars);
            return -1;
        }

        std::ostringstream oss;
        
        // Anonymous constructor syntax: ⟨field1, field2, ...⟩
        oss << "⟨" << r1cs->n_constraints
            << ", " << r1cs->n_vars
            << ", " << r1cs->n_public_inputs
            << ", " << params->modulus
            << ", " << sparse_matrix_to_lean(r1cs->A)
            << ", " << sparse_matrix_to_lean(r1cs->B)
            << ", " << sparse_matrix_to_lean(r1cs->C)
            << "⟩";
        
        std::string result = oss.str();
        
        // Copy to output buffer with null terminator
        if (result.size() + 1 > buffer_size) {
            fprintf(stderr, "export_vk_to_lean: buffer too small (need %zu, have %zu)\n",
                    result.size() + 1, buffer_size);
            return -1;
        }
        
        std::strcpy(out_buffer, result.c_str());
        return static_cast<int>(result.size());
        
    } catch (const std::exception& e) {
        fprintf(stderr, "export_vk_to_lean error: %s\n", e.what());
        return -1;
    }
}

/**
 * @brief Export security parameters to Lean record format.
 * 
 * Format: { n := 4096, k := 2, q := 12289, σ := 3.2, λ := 128 }
 * 
 * @param params Security parameters
 * @param out_buffer Output buffer (allocated by caller)
 * @param buffer_size Size of output buffer
 * @return Number of bytes written, or -1 on error
 */
int export_params_to_lean(
    const PublicParams* params,
    char* out_buffer,
    size_t buffer_size
) noexcept {
    if (!params || !out_buffer) return -1;
    
    try {
        std::string result = public_params_to_lean(params);
        
        if (result.size() + 1 > buffer_size) {
            return -1;
        }
        
        std::strcpy(out_buffer, result.c_str());
        return static_cast<int>(result.size());
        
    } catch (const std::exception& e) {
        fprintf(stderr, "export_params_to_lean error: %s\n", e.what());
        return -1;
    }
}

/**
 * @brief Export SEAL context to Lean format (EXPERIMENTAL).
 * 
 * Attempts to serialize SEAL EncryptionParameters for formal verification.
 * 
 * @param ctx LWE context (must contain SEAL context)
 * @param out_buffer Output buffer
 * @param buffer_size Size of output buffer
 * @return Number of bytes written, or -1 on error
 * 
 * @warning SEAL internals may not be fully serializable.
 *          This is a best-effort prototype for M10.2 validation.
 */
int export_seal_context_to_lean(
    const LweContext* ctx,
    char* out_buffer,
    size_t buffer_size
) noexcept {
    if (!ctx || !out_buffer) return -1;
    
    try {
        if (!ctx->seal_ctx) {
            fprintf(stderr, "export_seal_context_to_lean: SEAL context not initialized\n");
            return -1;
        }
        
        // Extract encryption parameters
        auto params = ctx->seal_ctx->key_context_data()->parms();
        std::string result = seal_params_to_lean(params);
        
        if (result.size() + 1 > buffer_size) {
            return -1;
        }
        
        std::strcpy(out_buffer, result.c_str());
        return static_cast<int>(result.size());
        
    } catch (const std::exception& e) {
        fprintf(stderr, "export_seal_context_to_lean error: %s\n", e.what());
        return -1;
    }
}

/**
 * @brief Export SEAL PublicKey to hex-encoded format (EXPERIMENTAL).
 * 
 * Serializes public key for Lean-side commitment verification.
 * 
 * @param ctx LWE context (must contain public key)
 * @param out_buffer Output buffer (large, ~1MB for typical keys)
 * @param buffer_size Size of output buffer
 * @return Number of bytes written, or -1 on error
 * 
 * @note Production version needs proper base64 encoding.
 */
int export_seal_pubkey_to_lean(
    const LweContext* ctx,
    char* out_buffer,
    size_t buffer_size
) noexcept {
    if (!ctx || !out_buffer) return -1;
    
    try {
        if (!ctx->seal_ctx || !ctx->pk) {
            fprintf(stderr, "export_seal_pubkey_to_lean: SEAL context or key not initialized\n");
            return -1;
        }
        
        std::string hex_key = seal_pubkey_to_base64(*ctx->pk, *ctx->seal_ctx);
        
        if (hex_key.empty() || hex_key.size() + 1 > buffer_size) {
            fprintf(stderr, "export_seal_pubkey_to_lean: serialization failed or buffer too small\n");
            return -1;
        }
        
        std::strcpy(out_buffer, hex_key.c_str());
        return static_cast<int>(hex_key.size());
        
    } catch (const std::exception& e) {
        fprintf(stderr, "export_seal_pubkey_to_lean error: %s\n", e.what());
        return -1;
    }
}

} // extern "C"
