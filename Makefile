.PHONY: help build test lint clean setup docs

# Default target
.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
RESET := \033[0m

help: ## Show this help message
	@echo "$(CYAN)ΛSNARK-R Makefile$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

setup: ## Install dependencies and setup development environment
	@echo "$(CYAN)Setting up development environment...$(RESET)"
	@# Install pre-commit hooks
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
	else \
		echo "Warning: pre-commit not found. Install with: pip install pre-commit"; \
	fi
	@# Install Rust tools
	@rustup component add clippy rustfmt
	@cargo install cargo-fuzz cargo-criterion || true
	@echo "$(GREEN)✓ Setup complete$(RESET)"

build: ## Build C++ core and Rust API
	@echo "$(CYAN)Building C++ core...$(RESET)"
	@cd cpp-core && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
	@echo "$(CYAN)Building Rust API...$(RESET)"
	@cd rust-api && cargo build --release
	@echo "$(GREEN)✓ Build complete$(RESET)"

build-dev: ## Build in debug mode
	@cd cpp-core && cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build
	@cd rust-api && cargo build

test: ## Run all tests
	@echo "$(CYAN)Running C++ tests...$(RESET)"
	@cd cpp-core/build && ctest --output-on-failure || true
	@echo "$(CYAN)Running Rust tests...$(RESET)"
	@cd rust-api && cargo test --all
	@echo "$(GREEN)✓ Tests complete$(RESET)"

test-cpp: ## Run C++ tests only
	@cd cpp-core/build && ctest --output-on-failure

test-rust: ## Run Rust tests only
	@cd rust-api && cargo test --all

test-ct: ## Run constant-time validation (dudect)
	@echo "$(CYAN)Running constant-time checks...$(RESET)"
	@cd rust-api && cargo bench --bench dudect -- --test
	@echo "$(GREEN)✓ Constant-time validation complete$(RESET)"

bench: ## Run benchmarks
	@echo "$(CYAN)Running benchmarks...$(RESET)"
	@cd rust-api && cargo bench --all
	@echo "$(GREEN)✓ Benchmarks complete$(RESET)"

lint: ## Run linters (clippy, clang-tidy)
	@echo "$(CYAN)Running Rust linters...$(RESET)"
	@cd rust-api && cargo clippy --all-targets --all-features -- -D warnings
	@cd rust-api && cargo fmt --all -- --check
	@echo "$(CYAN)Running C++ linters...$(RESET)"
	@# clang-tidy check (requires compilation database)
	@if [ -f cpp-core/build/compile_commands.json ]; then \
		cd cpp-core && clang-tidy src/*.cpp -p build; \
	else \
		echo "Warning: compile_commands.json not found. Run build first."; \
	fi
	@echo "$(GREEN)✓ Lint complete$(RESET)"

fmt: ## Format code
	@echo "$(CYAN)Formatting Rust code...$(RESET)"
	@cd rust-api && cargo fmt --all
	@echo "$(CYAN)Formatting C++ code...$(RESET)"
	@find cpp-core/src cpp-core/include -name '*.cpp' -o -name '*.h' | xargs clang-format -i
	@echo "$(GREEN)✓ Format complete$(RESET)"

clean: ## Clean build artifacts
	@echo "$(CYAN)Cleaning...$(RESET)"
	@rm -rf cpp-core/build
	@cd rust-api && cargo clean
	@rm -rf site/
	@echo "$(GREEN)✓ Clean complete$(RESET)"

docs: ## Build documentation
	@echo "$(CYAN)Building documentation...$(RESET)"
	@cd rust-api && cargo doc --no-deps --all-features
	@mkdocs build
	@echo "$(GREEN)✓ Documentation built$(RESET)"
	@echo "  Rust docs: rust-api/target/doc/lambda_snark/index.html"
	@echo "  Project docs: site/index.html"

docs-serve: ## Serve documentation locally
	@mkdocs serve

fuzz: ## Run fuzzer (1 hour)
	@cd rust-api/lambda-snark && cargo fuzz run fuzz_verify -- -max_total_time=3600

install: build ## Install library system-wide (requires sudo)
	@cd cpp-core/build && sudo cmake --install .
	@cd rust-api && cargo install --path lambda-snark-cli

.PHONY: ci ci-cpp ci-rust
ci: ci-cpp ci-rust ## Run full CI pipeline locally

ci-cpp: ## CI for C++ (build + test + lint)
	@cd cpp-core && cmake -B build -DCMAKE_BUILD_TYPE=Release -DLAMBDA_SNARK_BUILD_TESTS=ON
	@cd cpp-core/build && cmake --build . && ctest --output-on-failure

ci-rust: ## CI for Rust (build + test + lint)
	@cd rust-api && cargo build --all-features
	@cd rust-api && cargo test --all
	@cd rust-api && cargo clippy --all-targets -- -D warnings
	@cd rust-api && cargo fmt --all -- --check
