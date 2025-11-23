use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=../../cpp-core/include");
    println!("cargo:rerun-if-changed=../../cpp-core/src");
    println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
    println!("cargo:rerun-if-env-changed=SEAL_DIR");

    // Link SEAL (from vcpkg) - check environment variable or default location
    let vcpkg_root = env::var("VCPKG_ROOT").unwrap_or_else(|_| {
        if PathBuf::from("/home/kirill/vcpkg").exists() {
            "/home/kirill/vcpkg".to_string()
        } else {
            "../../vcpkg".to_string()
        }
    });

    // Build C++ core with CMake (vcpkg toolchain if available)
    let mut config = cmake::Config::new("../../cpp-core");
    config
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("LAMBDA_SNARK_BUILD_TESTS", "OFF")
        .define("LAMBDA_SNARK_BUILD_BENCHMARKS", "OFF");

    let toolchain = PathBuf::from(&vcpkg_root)
        .join("scripts")
        .join("buildsystems")
        .join("vcpkg.cmake");
    if toolchain.exists() {
        config
            .define(
                "CMAKE_TOOLCHAIN_FILE",
                toolchain.to_string_lossy().to_string(),
            )
            .define("VCPKG_TARGET_TRIPLET", "x64-linux")
            .define("VCPKG_FEATURE_FLAGS", "manifests")
            .env("VCPKG_ROOT", &vcpkg_root);
    }

    let seal_dir = PathBuf::from(&vcpkg_root)
        .join("installed")
        .join("x64-linux")
        .join("share")
        .join("seal");
    if seal_dir.exists() {
        config.define("SEAL_DIR", seal_dir.to_string_lossy().to_string());
    }

    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());

    let cmake_vcpkg = dst
        .join("build")
        .join("vcpkg_installed")
        .join("x64-linux")
        .join("lib");
    if cmake_vcpkg.exists() {
        println!("cargo:rustc-link-search=native={}", cmake_vcpkg.display());
    }

    println!("cargo:rustc-link-lib=static=lambda_snark_core");
    let vcpkg_lib = PathBuf::from(format!("{}/installed/x64-linux/lib", vcpkg_root));
    if vcpkg_lib.exists() {
        println!("cargo:rustc-link-search=native={}", vcpkg_lib.display());
        println!("cargo:rustc-link-lib=static=seal-4.1");
        println!("cargo:rustc-link-lib=static=zstd");
    } else {
        eprintln!(
            "Warning: vcpkg SEAL library not found at {}, using system libraries only",
            vcpkg_lib.display()
        );
    }
    println!("cargo:rustc-link-lib=dylib=z"); // system zlib
    println!("cargo:rustc-link-lib=dylib=ntl"); // system NTL
    println!("cargo:rustc-link-lib=dylib=gmp"); // system GMP (NTL dependency)
    println!("cargo:rustc-link-lib=dylib=pthread");
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=m");
    }

    // Link C++ standard library
    let target = env::var("TARGET").unwrap();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("linux") || target.contains("bsd") {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Generate Rust bindings
    let bindings = bindgen::Builder::default()
        .header("../../cpp-core/include/lambda_snark/types.h")
        .header("../../cpp-core/include/lambda_snark/commitment.h")
        .header("../../cpp-core/include/lambda_snark/ntt.h")
        .header("../../cpp-core/include/lambda_snark/r1cs.h")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .clang_arg("-I../../cpp-core/include")
        .clang_arg("-I/usr/include/c++/14")
        .clang_arg("-I/usr/include/x86_64-linux-gnu/c++/14")
        .clang_arg("-I/usr/lib/gcc/x86_64-linux-gnu/14/include")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("lwe_.*")
        .allowlist_function("ntt_.*")
        .allowlist_function("lambda_snark_r1cs_.*")
        .allowlist_type("Lwe.*")
        .allowlist_type("Ntt.*")
        .allowlist_type("R1CSConstraintSystem")
        .allowlist_type("SparseMatrix")
        .allowlist_type("SparseEntry")
        .allowlist_type("PublicParams")
        .allowlist_type("ProfileType")
        .allowlist_type("LambdaSnarkError")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
