use lambda_snark::arith::{add_mod, mod_inverse, mod_pow, sub_mod};
use lambda_snark::{Field, Polynomial, SparseMatrix};
use lambda_snark_core::NTT_MODULUS;
use rand::RngCore;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const SAMPLE_COUNT: usize = 20_000;

#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    mean: f64,
    variance: f64,
    count: usize,
}

fn compute_moments(samples: &[f64]) -> Moments {
    if samples.is_empty() {
        return Moments::default();
    }

    let count = samples.len();
    let sum: f64 = samples.iter().copied().sum();
    let mean = sum / count as f64;

    let mut var_acc = 0.0;
    for &value in samples {
        let diff = value - mean;
        var_acc += diff * diff;
    }

    let variance = if count > 1 {
        var_acc / (count as f64 - 1.0)
    } else {
        0.0
    };

    Moments {
        mean,
        variance,
        count,
    }
}

fn welch_t_stat(a: Moments, b: Moments) -> f64 {
    if a.count == 0 || b.count == 0 {
        return 0.0;
    }

    let denom = (a.variance / a.count as f64 + b.variance / b.count as f64).sqrt();
    if denom == 0.0 {
        return 0.0;
    }

    (a.mean - b.mean) / denom
}

fn locate_repo_root() -> PathBuf {
    let mut current = std::env::current_dir().expect("cwd");
    for _ in 0..6 {
        let readme = current.join("README.md");
        let security = current.join("SECURITY.md");
        if readme.exists() && security.exists() {
            return current;
        }
        if !current.pop() {
            break;
        }
    }
    std::env::current_dir().expect("cwd fallback")
}

fn measure_mod_pow(rng: &mut ChaCha20Rng) -> (Moments, Moments, f64) {
    let mut class_zero = Vec::with_capacity(SAMPLE_COUNT / 2 + 1);
    let mut class_one = Vec::with_capacity(SAMPLE_COUNT / 2 + 1);

    for _ in 0..SAMPLE_COUNT {
        let class = (rng.next_u32() & 1) == 1;
        let base = rng.next_u64() % NTT_MODULUS;
        let mut exp = rng.next_u64();

        if class {
            exp |= 1; // ensure odd exponent
        } else {
            exp &= !1; // force even exponent
        }

        let start = Instant::now();
        let _ = mod_pow(base, exp, NTT_MODULUS);
        let elapsed = start.elapsed().as_nanos() as f64;

        if class {
            class_one.push(elapsed);
        } else {
            class_zero.push(elapsed);
        }
    }

    let zero = compute_moments(&class_zero);
    let one = compute_moments(&class_one);
    let t = welch_t_stat(zero, one);

    (zero, one, t)
}

fn sample_inverse_input(rng: &mut ChaCha20Rng, class_one: bool) -> u64 {
    loop {
        if class_one {
            let candidate = (rng.next_u64() % (NTT_MODULUS - 1)) + 1;
            if candidate != 0 {
                return candidate;
            }
        } else {
            let candidate = ((rng.next_u64() % 1_048_576) + 1) as u64; // keep small
            if candidate != 0 {
                return candidate;
            }
        }
    }
}

fn sample_add_inputs(rng: &mut ChaCha20Rng, expect_wrap: bool) -> (u64, u64) {
    loop {
        let a = rng.next_u64() % NTT_MODULUS;
        let b = rng.next_u64() % NTT_MODULUS;

        let wraps = match a.checked_add(b) {
            Some(sum) => sum >= NTT_MODULUS,
            None => true,
        };

        if wraps == expect_wrap {
            return (a, b);
        }
    }
}

fn sample_sub_inputs(rng: &mut ChaCha20Rng, expect_wrap: bool) -> (u64, u64) {
    loop {
        let a = rng.next_u64() % NTT_MODULUS;
        let b = rng.next_u64() % NTT_MODULUS;
        let wraps = a < b;
        if wraps == expect_wrap {
            return (a, b);
        }
    }
}

fn measure_mod_inverse(rng: &mut ChaCha20Rng) -> (Moments, Moments, f64) {
    let mut class_zero = Vec::with_capacity(SAMPLE_COUNT / 2 + 1);
    let mut class_one = Vec::with_capacity(SAMPLE_COUNT / 2 + 1);

    for _ in 0..SAMPLE_COUNT {
        let class = (rng.next_u32() & 1) == 1;
        let input = sample_inverse_input(rng, class);

        let start = Instant::now();
        let _ = mod_inverse(input, NTT_MODULUS).expect("invertible");
        let elapsed = start.elapsed().as_nanos() as f64;

        if class {
            class_one.push(elapsed);
        } else {
            class_zero.push(elapsed);
        }
    }

    let zero = compute_moments(&class_zero);
    let one = compute_moments(&class_one);
    let t = welch_t_stat(zero, one);

    (zero, one, t)
}

fn measure_add_mod(rng: &mut ChaCha20Rng) -> (Moments, Moments, f64) {
    let mut class_zero = Vec::with_capacity(SAMPLE_COUNT / 2 + 1);
    let mut class_one = Vec::with_capacity(SAMPLE_COUNT / 2 + 1);

    for _ in 0..SAMPLE_COUNT {
        let class = (rng.next_u32() & 1) == 1;
        let (a, b) = sample_add_inputs(rng, class);

        let start = Instant::now();
        let _ = add_mod(a, b, NTT_MODULUS);
        let elapsed = start.elapsed().as_nanos() as f64;

        if class {
            class_one.push(elapsed);
        } else {
            class_zero.push(elapsed);
        }
    }

    let zero = compute_moments(&class_zero);
    let one = compute_moments(&class_one);
    let t = welch_t_stat(zero, one);

    (zero, one, t)
}

fn measure_sub_mod(rng: &mut ChaCha20Rng) -> (Moments, Moments, f64) {
    let mut class_zero = Vec::with_capacity(SAMPLE_COUNT / 2 + 1);
    let mut class_one = Vec::with_capacity(SAMPLE_COUNT / 2 + 1);

    for _ in 0..SAMPLE_COUNT {
        let class = (rng.next_u32() & 1) == 1;
        let (a, b) = sample_sub_inputs(rng, class);

        let start = Instant::now();
        let _ = sub_mod(a, b, NTT_MODULUS);
        let elapsed = start.elapsed().as_nanos() as f64;

        if class {
            class_one.push(elapsed);
        } else {
            class_zero.push(elapsed);
        }
    }

    let zero = compute_moments(&class_zero);
    let one = compute_moments(&class_one);
    let t = welch_t_stat(zero, one);

    (zero, one, t)
}

fn generate_polynomials(rng: &mut ChaCha20Rng) -> Vec<Polynomial> {
    const POLY_COUNT: usize = 8;
    const DEGREE: usize = 15;

    let mut polys = Vec::with_capacity(POLY_COUNT);
    for _ in 0..POLY_COUNT {
        let coeffs: Vec<u64> = (0..=DEGREE).map(|_| rng.next_u64() % NTT_MODULUS).collect();
        polys.push(Polynomial::from_witness(&coeffs, NTT_MODULUS));
    }

    polys
}

fn measure_polynomial_evaluate(rng: &mut ChaCha20Rng) -> (Moments, Moments, f64) {
    let polynomials = generate_polynomials(rng);
    let poly_len = polynomials.len();

    let mut class_zero = Vec::with_capacity(SAMPLE_COUNT / 2 + 1);
    let mut class_one = Vec::with_capacity(SAMPLE_COUNT / 2 + 1);

    for _ in 0..SAMPLE_COUNT {
        let class = (rng.next_u32() & 1) == 1;
        let index = (rng.next_u32() as usize) % poly_len;
        let poly = &polynomials[index];

        let alpha = if class {
            // Force higher magnitude evaluation point.
            (rng.next_u64() % (NTT_MODULUS / 2)) + (NTT_MODULUS / 2)
        } else {
            rng.next_u64() % (NTT_MODULUS / 2)
        };

        let start = Instant::now();
        let _ = poly.evaluate(Field::new(alpha % NTT_MODULUS));
        let elapsed = start.elapsed().as_nanos() as f64;

        if class {
            class_one.push(elapsed);
        } else {
            class_zero.push(elapsed);
        }
    }

    let zero = compute_moments(&class_zero);
    let one = compute_moments(&class_one);
    let t = welch_t_stat(zero, one);

    (zero, one, t)
}

fn build_sparse_matrix(rng: &mut ChaCha20Rng) -> SparseMatrix {
    const DIM: usize = 16;

    let dense: Vec<Vec<u64>> = (0..DIM)
        .map(|_| {
            (0..DIM)
                .map(|_| rng.next_u64() % NTT_MODULUS)
                .collect::<Vec<u64>>()
        })
        .collect();

    SparseMatrix::from_dense(&dense)
}

fn measure_sparse_mul(rng: &mut ChaCha20Rng) -> (Moments, Moments, f64) {
    let matrix = build_sparse_matrix(rng);

    let vec_low: Vec<u64> = (0..matrix.cols())
        .map(|_| rng.next_u64() % (NTT_MODULUS / 8))
        .collect();
    let vec_high: Vec<u64> = (0..matrix.cols())
        .map(|_| (rng.next_u64() % (NTT_MODULUS / 8)) + (NTT_MODULUS - (NTT_MODULUS / 8)))
        .collect();

    let mut class_zero = Vec::with_capacity(SAMPLE_COUNT / 2 + 1);
    let mut class_one = Vec::with_capacity(SAMPLE_COUNT / 2 + 1);

    for _ in 0..SAMPLE_COUNT {
        let class = (rng.next_u32() & 1) == 1;
        let vector = if class { &vec_high } else { &vec_low };

        let start = Instant::now();
        let _ = matrix.mul_vec(vector, NTT_MODULUS);
        let elapsed = start.elapsed().as_nanos() as f64;

        if class {
            class_one.push(elapsed);
        } else {
            class_zero.push(elapsed);
        }
    }

    let zero = compute_moments(&class_zero);
    let one = compute_moments(&class_one);
    let t = welch_t_stat(zero, one);

    (zero, one, t)
}

fn write_report(
    path: &Path,
    add_stats: (Moments, Moments, f64),
    sub_stats: (Moments, Moments, f64),
    mod_pow_stats: (Moments, Moments, f64),
    mod_inv_stats: (Moments, Moments, f64),
    poly_eval_stats: (Moments, Moments, f64),
    sparse_mul_stats: (Moments, Moments, f64),
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(path)?;

    let (add_zero, add_one, add_t) = add_stats;
    let (sub_zero, sub_one, sub_t) = sub_stats;
    let (pow_zero, pow_one, pow_t) = mod_pow_stats;
    let (inv_zero, inv_one, inv_t) = mod_inv_stats;
    let (poly_zero, poly_one, poly_t) = poly_eval_stats;
    let (sparse_zero, sparse_one, sparse_t) = sparse_mul_stats;

    writeln!(file, "# dudect modular arithmetic report\n")?;

    writeln!(file, "## add_mod\n")?;
    writeln!(file, "- samples_per_class_zero: {}", add_zero.count)?;
    writeln!(file, "- samples_per_class_one: {}", add_one.count)?;
    writeln!(file, "- total_samples: {}", add_zero.count + add_one.count)?;
    writeln!(file, "- welch_t_statistic: {:.4}\n", add_t)?;
    writeln!(file, "| class | samples | mean_ns | stddev_ns |")?;
    writeln!(file, "|-------|---------|---------|-----------|")?;
    writeln!(
        file,
        "| 0 | {} | {:.2} | {:.2} |",
        add_zero.count,
        add_zero.mean,
        add_zero.variance.sqrt()
    )?;
    writeln!(
        file,
        "| 1 | {} | {:.2} | {:.2} |\n",
        add_one.count,
        add_one.mean,
        add_one.variance.sqrt()
    )?;

    writeln!(file, "## sub_mod\n")?;
    writeln!(file, "- samples_per_class_zero: {}", sub_zero.count)?;
    writeln!(file, "- samples_per_class_one: {}", sub_one.count)?;
    writeln!(file, "- total_samples: {}", sub_zero.count + sub_one.count)?;
    writeln!(file, "- welch_t_statistic: {:.4}\n", sub_t)?;
    writeln!(file, "| class | samples | mean_ns | stddev_ns |")?;
    writeln!(file, "|-------|---------|---------|-----------|")?;
    writeln!(
        file,
        "| 0 | {} | {:.2} | {:.2} |",
        sub_zero.count,
        sub_zero.mean,
        sub_zero.variance.sqrt()
    )?;
    writeln!(
        file,
        "| 1 | {} | {:.2} | {:.2} |\n",
        sub_one.count,
        sub_one.mean,
        sub_one.variance.sqrt()
    )?;

    writeln!(file, "## mod_pow\n")?;
    writeln!(file, "- samples_per_class_zero: {}", pow_zero.count)?;
    writeln!(file, "- samples_per_class_one: {}", pow_one.count)?;
    writeln!(file, "- total_samples: {}", pow_zero.count + pow_one.count)?;
    writeln!(file, "- welch_t_statistic: {:.4}\n", pow_t)?;
    writeln!(file, "| class | samples | mean_ns | stddev_ns |")?;
    writeln!(file, "|-------|---------|---------|-----------|")?;
    writeln!(
        file,
        "| 0 | {} | {:.2} | {:.2} |",
        pow_zero.count,
        pow_zero.mean,
        pow_zero.variance.sqrt()
    )?;
    writeln!(
        file,
        "| 1 | {} | {:.2} | {:.2} |\n",
        pow_one.count,
        pow_one.mean,
        pow_one.variance.sqrt()
    )?;

    writeln!(file, "## mod_inverse\n")?;
    writeln!(file, "- samples_per_class_zero: {}", inv_zero.count)?;
    writeln!(file, "- samples_per_class_one: {}", inv_one.count)?;
    writeln!(file, "- total_samples: {}", inv_zero.count + inv_one.count)?;
    writeln!(file, "- welch_t_statistic: {:.4}\n", inv_t)?;
    writeln!(file, "| class | samples | mean_ns | stddev_ns |")?;
    writeln!(file, "|-------|---------|---------|-----------|")?;
    writeln!(
        file,
        "| 0 | {} | {:.2} | {:.2} |",
        inv_zero.count,
        inv_zero.mean,
        inv_zero.variance.sqrt()
    )?;
    writeln!(
        file,
        "| 1 | {} | {:.2} | {:.2} |\n",
        inv_one.count,
        inv_one.mean,
        inv_one.variance.sqrt()
    )?;

    writeln!(file, "## polynomial_evaluate\n")?;
    writeln!(file, "- samples_per_class_zero: {}", poly_zero.count)?;
    writeln!(file, "- samples_per_class_one: {}", poly_one.count)?;
    writeln!(
        file,
        "- total_samples: {}",
        poly_zero.count + poly_one.count
    )?;
    writeln!(file, "- welch_t_statistic: {:.4}\n", poly_t)?;
    writeln!(file, "| class | samples | mean_ns | stddev_ns |")?;
    writeln!(file, "|-------|---------|---------|-----------|")?;
    writeln!(
        file,
        "| 0 | {} | {:.2} | {:.2} |",
        poly_zero.count,
        poly_zero.mean,
        poly_zero.variance.sqrt()
    )?;
    writeln!(
        file,
        "| 1 | {} | {:.2} | {:.2} |\n",
        poly_one.count,
        poly_one.mean,
        poly_one.variance.sqrt()
    )?;

    writeln!(file, "## sparse_matrix_mul_vec\n")?;
    writeln!(file, "- samples_per_class_zero: {}", sparse_zero.count)?;
    writeln!(file, "- samples_per_class_one: {}", sparse_one.count)?;
    writeln!(
        file,
        "- total_samples: {}",
        sparse_zero.count + sparse_one.count
    )?;
    writeln!(file, "- welch_t_statistic: {:.4}\n", sparse_t)?;
    writeln!(file, "| class | samples | mean_ns | stddev_ns |")?;
    writeln!(file, "|-------|---------|---------|-----------|")?;
    writeln!(
        file,
        "| 0 | {} | {:.2} | {:.2} |",
        sparse_zero.count,
        sparse_zero.mean,
        sparse_zero.variance.sqrt()
    )?;
    writeln!(
        file,
        "| 1 | {} | {:.2} | {:.2} |\n",
        sparse_one.count,
        sparse_one.mean,
        sparse_one.variance.sqrt()
    )?;

    writeln!(
        file,
        "> Threshold guidance: |t| < 4.5 is typically treated as constant-time by dudect."
    )?;

    Ok(())
}

fn main() -> std::io::Result<()> {
    let mut rng = ChaCha20Rng::seed_from_u64(0x54494D45); // "TIME"

    let add_stats = measure_add_mod(&mut rng);
    let sub_stats = measure_sub_mod(&mut rng);
    let mod_pow_stats = measure_mod_pow(&mut rng);
    let mod_inverse_stats = measure_mod_inverse(&mut rng);
    let poly_eval_stats = measure_polynomial_evaluate(&mut rng);
    let sparse_mul_stats = measure_sparse_mul(&mut rng);

    let repo_root = locate_repo_root();
    let output_path = repo_root
        .join("artifacts")
        .join("dudect")
        .join("mod_arith_report.md");
    write_report(
        &output_path,
        add_stats,
        sub_stats,
        mod_pow_stats,
        mod_inverse_stats,
        poly_eval_stats,
        sparse_mul_stats,
    )?;

    println!(
        "add_mod t-stat: {:.4}, sub_mod t-stat: {:.4}, mod_pow t-stat: {:.4}, mod_inverse t-stat: {:.4}, poly_eval t-stat: {:.4}, sparse_mul t-stat: {:.4} (report: {})",
        add_stats.2,
        sub_stats.2,
        mod_pow_stats.2,
        mod_inverse_stats.2,
        poly_eval_stats.2,
        sparse_mul_stats.2,
        output_path.display()
    );

    Ok(())
}
