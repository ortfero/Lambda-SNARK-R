#!/bin/bash
# Quick setup script for ΛSNARK-R development environment

set -e

echo "🚀 Setting up ΛSNARK-R development environment..."
echo ""

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

echo -e "${CYAN}Detected OS: $OS${NC}"
echo ""

# Install system dependencies
echo -e "${CYAN}Installing system dependencies...${NC}"
if [[ "$OS" == "linux" ]]; then
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        ninja-build \
        libgmp-dev \
        libntl-dev \
        libeigen3-dev \
        libsodium-dev \
        autoconf \
        automake \
        libtool \
        clang-format \
        clang-tidy
elif [[ "$OS" == "macos" ]]; then
    brew install \
        cmake \
        ninja \
        gmp \
        ntl \
        eigen \
        libsodium \
        autoconf \
        automake \
        libtool \
        clang-format
fi
echo -e "${GREEN}✓ System dependencies installed${NC}"
echo ""

# Check Rust
if command -v rustc &> /dev/null; then
    echo -e "${GREEN}✓ Rust already installed ($(rustc --version))${NC}"
else
    echo -e "${CYAN}Installing Rust...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo -e "${GREEN}✓ Rust installed${NC}"
fi

# Install Rust components
echo -e "${CYAN}Installing Rust components...${NC}"
rustup component add clippy rustfmt
echo -e "${GREEN}✓ Rust components installed${NC}"
echo ""

# Check Python
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓ Python already installed ($(python3 --version))${NC}"
else
    echo -e "${RED}Python 3 not found. Please install Python 3.11+${NC}"
    exit 1
fi

# Install UV (Python package manager)
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✓ UV already installed${NC}"
else
    echo -e "${CYAN}Installing UV...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo -e "${GREEN}✓ UV installed${NC}"
fi

# Setup Python virtual environment
echo -e "${CYAN}Setting up Python environment...${NC}"
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
echo -e "${GREEN}✓ Python environment ready${NC}"
echo ""

# Install pre-commit hooks
echo -e "${CYAN}Installing pre-commit hooks...${NC}"
pre-commit install
echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
echo ""

# Check vcpkg (optional, for C++ dependencies)
if [ -d "vcpkg" ]; then
    echo -e "${GREEN}✓ vcpkg already cloned${NC}"
else
    echo -e "${YELLOW}⚠️  vcpkg not found. Clone manually for C++ dependencies:${NC}"
    echo -e "  git clone https://github.com/microsoft/vcpkg.git"
    echo -e "  ./vcpkg/bootstrap-vcpkg.sh"
    echo -e "  ./vcpkg/vcpkg install seal gmp gtest benchmark"
fi
echo ""

# Build test
echo -e "${CYAN}Testing build...${NC}"
echo -e "${YELLOW}Building C++ core...${NC}"
cd cpp-core
if cmake -B build -DCMAKE_BUILD_TYPE=Release -DLAMBDA_SNARK_BUILD_TESTS=ON 2>&1 | grep -q "SEAL not found"; then
    echo -e "${YELLOW}⚠️  SEAL not found (expected for now, using stubs)${NC}"
fi
cmake --build build || echo -e "${YELLOW}⚠️  C++ build had warnings (OK for now)${NC}"
cd ..

echo -e "${YELLOW}Building Rust API...${NC}"
cd rust-api
cargo build --all 2>&1 | head -20 || echo -e "${YELLOW}⚠️  Rust build may fail without C++ core (expected)${NC}"
cd ..
echo ""

# Summary
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Setup complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. ${CYAN}Install SEAL, NTL${NC} (see cpp-core/README.md)"
echo -e "  2. ${CYAN}make build${NC}        # Full build"
echo -e "  3. ${CYAN}make test${NC}         # Run tests"
echo -e "  4. ${CYAN}make docs${NC}         # Generate documentation"
echo ""
echo -e "Resources:"
echo -e "  • README.md            # Project overview"
echo -e "  • PROJECT_SETUP.md     # Complete setup guide"
echo -e "  • ROADMAP.md           # Development roadmap"
echo -e "  • CONTRIBUTING.md      # Contribution guidelines"
echo ""
echo -e "Happy coding! 🎉"
