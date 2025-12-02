use std::env;
use std::path::{Path, PathBuf};

fn link_from_dir(lib_dir: &Path, lib: &str, fallback: &str) {
    let static_lib = lib_dir.join(format!("lib{}.a", lib));
    if static_lib.exists() {
        println!("cargo:rustc-link-lib=static={}", lib);
        return;
    }

    let shared_exts = ["so", "dylib", "dll"];
    if shared_exts
        .iter()
        .map(|ext| lib_dir.join(format!("lib{}.{}", lib, ext)))
        .any(|candidate| candidate.exists())
    {
        println!("cargo:rustc-link-lib=dylib={}", lib);
        return;
    }

    println!("cargo:rustc-link-lib={}", fallback);
}

fn main() {
    println!("cargo:rerun-if-changed=../../cpp-core/include");
    println!("cargo:rerun-if-changed=../../cpp-core/src");
    println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
    println!("cargo:rerun-if-env-changed=SEAL_DIR");
    println!("cargo:rerun-if-env-changed=VCPKG_TARGET_TRIPLET");

    // Link SEAL (from vcpkg) - check environment variable or default location
    let vcpkg_root = env::var("VCPKG_ROOT").unwrap_or_else(|_| {
        if PathBuf::from("/home/kirill/vcpkg").exists() {
            "/home/kirill/vcpkg".to_string()
        } else {
            "../../vcpkg".to_string()
        }
    });

    let target = env::var("TARGET").expect("TARGET not set by Cargo");
    // Pick triplet from env or infer from target triple
    let vcpkg_triplet = env::var("VCPKG_TARGET_TRIPLET").unwrap_or_else(|_| {
        if target.contains("apple-darwin") {
            if target.starts_with("aarch64") || target.starts_with("arm") {
                "arm64-osx".to_string()
            } else {
                "x64-osx".to_string()
            }
        } else if target.contains("windows") {
            "x64-windows".to_string()
        } else {
            "x64-linux".to_string()
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
            .define("VCPKG_TARGET_TRIPLET", &vcpkg_triplet)
            .define("VCPKG_FEATURE_FLAGS", "manifests")
            .env("VCPKG_ROOT", &vcpkg_root);
    }

    let seal_dir = PathBuf::from(&vcpkg_root)
        .join("installed")
        .join(&vcpkg_triplet)
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
        .join(&vcpkg_triplet)
        .join("lib");
    if cmake_vcpkg.exists() {
        println!("cargo:rustc-link-search=native={}", cmake_vcpkg.display());
    }

    println!("cargo:rustc-link-lib=static=lambda_snark_core");
    let vcpkg_lib = PathBuf::from(format!("{}/installed/{}/lib", vcpkg_root, vcpkg_triplet));
    let primary_lib_dir = if vcpkg_lib.exists() {
        Some(vcpkg_lib.clone())
    } else if cmake_vcpkg.exists() {
        Some(cmake_vcpkg.clone())
    } else {
        None
    };

    if let Some(lib_dir) = primary_lib_dir {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=static=seal-4.1");
        println!("cargo:rustc-link-lib=static=zstd");
        link_from_dir(&lib_dir, "ntl", "ntl");
        link_from_dir(&lib_dir, "gmp", "gmp");
    } else {
        eprintln!(
            "Warning: vcpkg SEAL library not found at {} or {}, using system libraries only",
            vcpkg_lib.display(),
            cmake_vcpkg.display()
        );
    }
    println!("cargo:rustc-link-lib=dylib=z"); // system zlib
    // If NTL/GMP were not found in vcpkg, fall back to common system locations (e.g., Homebrew)
    let mut have_ntl = vcpkg_lib.join("libntl.a").exists()
        || vcpkg_lib.join("libntl.dylib").exists()
        || vcpkg_lib.join("libntl.so").exists();
    let mut have_gmp = vcpkg_lib.join("libgmp.a").exists()
        || vcpkg_lib.join("libgmp.dylib").exists()
        || vcpkg_lib.join("libgmp.so").exists();

    let system_lib_prefixes = ["/opt/homebrew/lib", "/usr/local/lib", "/usr/lib"];
    for prefix in system_lib_prefixes {
        if !have_ntl {
            let candidate = Path::new(prefix).join("libntl.dylib");
            if candidate.exists() {
                println!(
                    "cargo:rustc-link-search=native={}",
                    Path::new(prefix).display()
                );
                println!("cargo:rustc-link-lib=dylib=ntl");
                have_ntl = true;
            }
        }
        if !have_gmp {
            let candidate = Path::new(prefix).join("libgmp.dylib");
            if candidate.exists() {
                println!(
                    "cargo:rustc-link-search=native={}",
                    Path::new(prefix).display()
                );
                println!("cargo:rustc-link-lib=dylib=gmp");
                have_gmp = true;
            }
        }
        if have_ntl && have_gmp {
            break;
        }
    }
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
    let mut bindings = bindgen::Builder::default()
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
        .allowlist_type("LambdaSnarkError");

    // Prefer vcpkg headers for NTL/SEAL if available
    let vcpkg_include = PathBuf::from(&vcpkg_root)
        .join("installed")
        .join(&vcpkg_triplet)
        .join("include");
    if vcpkg_include.exists() {
        bindings = bindings.clang_arg(format!("-I{}", vcpkg_include.display()));
    }

    // Fallback to common system prefixes for NTL if not present in vcpkg
    let ntl_header = "NTL/ZZ_p.h";
    let system_prefixes = [
        "/opt/homebrew/include",
        "/usr/local/include",
        "/usr/include",
    ];
    for prefix in system_prefixes {
        let candidate = Path::new(prefix).join(ntl_header);
        if candidate.exists() {
            bindings = bindings.clang_arg(format!("-I{}", Path::new(prefix).display()));
            break;
        }
    }

    let bindings = bindings.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
