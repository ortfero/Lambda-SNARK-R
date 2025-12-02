.PHONY: help build test lint clean setup docs

# Default target
.DEFAULT_GOAL := help

# VCPKG environment (adjust path to your installation)
VCPKG_ROOT ?= $(CURDIR)/vcpkg
# Expand $HOME or ~ if provided by the user, now that the default is a plain path
VCPKG_ROOT := $(shell sh -c 'eval echo "$$1"' sh "$(VCPKG_ROOT)")
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    VCPKG_TRIPLET ?= arm64-osx
  else
    VCPKG_TRIPLET ?= x64-osx
  endif
else
  VCPKG_TRIPLET ?= x64-linux
endif
export VCPKG_ROOT
VCPKG_TOOLCHAIN := $(VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake
VCPKG_BIN := $(VCPKG_ROOT)/vcpkg

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
	@# Check autotools needed for gmp via vcpkg
	@missing_autotools=""; \
	for tool in autoconf automake libtool; do \
		if ! command -v $$tool >/dev/null 2>&1; then \
			missing_autotools="$$missing_autotools $$tool"; \
		fi; \
	done; \
	if [ -n "$$missing_autotools" ]; then \
		echo "Missing tools:$$missing_autotools (required for gmp build via vcpkg)."; \
		echo "Install on macOS (Homebrew): brew install autoconf automake libtool"; \
		echo "Install on Ubuntu/Debian:    sudo apt-get install autoconf automake libtool"; \
		echo "Install on Fedora/RHEL:      sudo dnf install autoconf automake libtool"; \
		exit 1; \
	fi
	@# Bootstrap project-local vcpkg if needed
	@if [ ! -d "$$VCPKG_ROOT" ]; then \
		echo "Cloning vcpkg into $$VCPKG_ROOT ..."; \
		git clone https://github.com/microsoft/vcpkg.git "$$VCPKG_ROOT"; \
	fi
	@if [ ! -x "$$VCPKG_ROOT/vcpkg" ]; then \
		echo "Bootstrapping vcpkg..."; \
		(cd "$$VCPKG_ROOT" && ./bootstrap-vcpkg.sh -disableMetrics); \
	fi
	@# Install C++ deps from manifest (cpp-core/vcpkg.json)
	@echo "Installing vcpkg dependencies (triplet=$(VCPKG_TRIPLET))..."
	@cd "$$VCPKG_ROOT" && ./vcpkg install --triplet $(VCPKG_TRIPLET) --x-manifest-root="$(CURDIR)/cpp-core" --feature-flags=manifests
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
	@if [ ! -x "$(VCPKG_BIN)" ]; then \
		echo "vcpkg not bootstrapped at $(VCPKG_ROOT). Run: make setup"; \
		exit 1; \
	fi
	@cd cpp-core && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$(VCPKG_TOOLCHAIN) && cmake --build build
	@echo "$(CYAN)Building Rust API...$(RESET)"
	@cd rust-api && cargo build --release
	@echo "$(GREEN)✓ Build complete$(RESET)"

build-dev: ## Build in debug mode
	@if [ ! -x "$(VCPKG_BIN)" ]; then \
		echo "vcpkg not bootstrapped at $(VCPKG_ROOT). Run: make setup"; \
		exit 1; \
	fi
	@cd cpp-core && cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=$(VCPKG_TOOLCHAIN) && cmake --build build
	@cd rust-api && cargo build

test: ## Run all tests
	@echo "$(CYAN)Running C++ tests...$(RESET)"
	@cd cpp-core/build && ctest --output-on-failure || true
	@echo "$(CYAN)Running Rust tests...$(RESET)"
	@cd rust-api && cargo test --all
	@echo "$(GREEN)✓ Tests complete$(RESET)"

test-cpp: ## Run C++ tests only
	@echo "$(CYAN)Running C++ tests...$(RESET)"
	@if [ ! -d cpp-core/build ]; then \
		echo "Configuring C++ build (Release)..."; \
		cd cpp-core && cmake -B build -DCMAKE_BUILD_TYPE=Release -DLAMBDA_SNARK_BUILD_TESTS=ON -DCMAKE_TOOLCHAIN_FILE=$(VCPKG_TOOLCHAIN); \
	fi
	@cd cpp-core/build && cmake --build . && ctest --output-on-failure

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
	@cd cpp-core && cmake -B build -DCMAKE_BUILD_TYPE=Release -DLAMBDA_SNARK_BUILD_TESTS=ON -DCMAKE_TOOLCHAIN_FILE=$(VCPKG_TOOLCHAIN)
	@cd cpp-core/build && cmake --build . && ctest --output-on-failure

ci-rust: ## CI for Rust (build + test + lint)
	@cd rust-api && cargo build --all-features
	@cd rust-api && cargo test --all
	@cd rust-api && cargo clippy --all-targets -- -D warnings
	@cd rust-api && cargo fmt --all -- --check
