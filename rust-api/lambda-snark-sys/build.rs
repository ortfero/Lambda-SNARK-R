use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=../../cpp-core/include");
    println!("cargo:rerun-if-changed=../../cpp-core/src");
    
    // Build C++ core with CMake
    let dst = cmake::Config::new("../../cpp-core")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("LAMBDA_SNARK_BUILD_TESTS", "OFF")
        .define("LAMBDA_SNARK_BUILD_BENCHMARKS", "OFF")
        .build();
    
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-lib=static=lambda_snark_core");
    
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
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("lwe_.*")
        .allowlist_function("ntt_.*")
        .allowlist_type("Lwe.*")
        .allowlist_type("Ntt.*")
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
