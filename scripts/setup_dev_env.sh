#!/bin/bash
# Developer Environment Setup Script for QubitPulseOpt
# This script automates the installation of development dependencies and pre-commit hooks

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "  $1"
}

# Check if running from project root
if [ ! -f ".pre-commit-config.yaml" ]; then
    print_error "This script must be run from the project root directory"
    print_info "Please cd to the QubitPulseOpt directory first"
    exit 1
fi

print_header "QubitPulseOpt Developer Environment Setup"

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python not found. Please install Python 3.9 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
print_info "Found Python: $PYTHON_VERSION"

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_warning "No active virtual environment detected"
    print_info "It's recommended to use a virtual environment"
    print_info ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Setup cancelled. Create and activate a venv first:"
        print_info "  python3 -m venv venv"
        print_info "  source venv/bin/activate"
        exit 0
    fi
else
    if [ -n "$VIRTUAL_ENV" ]; then
        print_success "Virtual environment active: $VIRTUAL_ENV"
    else
        print_success "Conda environment active: $CONDA_DEFAULT_ENV"
    fi
fi

# Install development dependencies
print_header "Installing Development Dependencies"

print_info "Installing pre-commit..."
$PYTHON_CMD -m pip install --quiet --upgrade pre-commit
print_success "pre-commit installed"

print_info "Installing code quality tools..."
$PYTHON_CMD -m pip install --quiet --upgrade black flake8 isort
print_success "Code quality tools installed"

print_info "Installing additional flake8 plugins..."
$PYTHON_CMD -m pip install --quiet --upgrade flake8-docstrings flake8-bugbear
print_success "Flake8 plugins installed"

print_info "Installing testing tools..."
$PYTHON_CMD -m pip install --quiet --upgrade pytest pytest-cov pytest-rerunfailures
print_success "Testing tools installed"

# Install pre-commit hooks
print_header "Installing Pre-commit Hooks"

print_info "Installing git hooks..."
pre-commit install
print_success "Git hooks installed"

# Optional: install commit-msg hook
read -p "Install commit message validation hook? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    pre-commit install --hook-type commit-msg
    print_success "Commit message hook installed"
fi

# Run pre-commit on sample files to cache environments
print_header "Setting Up Pre-commit Environments"

print_info "This may take a few minutes on first run..."
print_info "(Pre-commit is downloading and caching hook environments)"
echo ""

# Run on a small subset first to avoid long wait
if pre-commit run --files README.md 2>&1 | grep -q "Passed\|Fixed"; then
    print_success "Pre-commit environments initialized"
else
    print_warning "Pre-commit initialization completed (some hooks may have warnings)"
fi

# Offer to run on all files
echo ""
read -p "Run pre-commit on all files now? (This may take 2-5 minutes) (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Running pre-commit on all files..."
    if pre-commit run --all-files; then
        print_success "All files passed pre-commit checks!"
    else
        print_warning "Some files failed pre-commit checks"
        print_info "Auto-fixable issues have been corrected"
        print_info "Review changes with: git diff"
        print_info "You may need to fix remaining issues manually"
    fi
fi

# Summary
print_header "Setup Complete!"

echo ""
print_success "Development environment is ready"
echo ""
print_info "Installed tools:"
print_info "  • pre-commit   - Git hook framework"
print_info "  • black        - Code formatter"
print_info "  • isort        - Import sorter"
print_info "  • flake8       - Linter"
print_info "  • pytest       - Test framework"
echo ""
print_info "Next steps:"
print_info "  1. Make code changes as usual"
print_info "  2. Stage files: git add <files>"
print_info "  3. Commit: git commit -m 'Your message'"
print_info "     → Pre-commit hooks run automatically!"
echo ""
print_info "Useful commands:"
print_info "  • pre-commit run --all-files    - Run all hooks manually"
print_info "  • pre-commit run black           - Run specific hook"
print_info "  • SKIP=hook git commit           - Skip specific hook"
print_info "  • git commit --no-verify         - Skip all hooks (use sparingly!)"
echo ""
print_info "Documentation:"
print_info "  • docs/DEVELOPER_GUIDE_PRECOMMIT.md - Full developer guide"
print_info "  • tests/README_TESTING.md           - Testing guide"
echo ""
print_success "Happy coding! 🚀"
