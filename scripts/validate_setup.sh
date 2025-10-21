#!/bin/bash
# Validation script for QubitPulseOpt setup (SOW Week 1.1)

echo "=== QubitPulseOpt Setup Validation ==="
echo ""

# Check 1: Git repository
echo "[1/5] Checking Git repository..."
if [ -d .git ]; then
    echo "✓ Git initialized"
    git log --oneline -1
else
    echo "✗ Git not initialized"
    exit 1
fi
echo ""

# Check 2: Directory structure
echo "[2/5] Checking directory structure..."
REQUIRED_DIRS=("src/hamiltonian" "src/pulses" "src/optimization" "src/noise" "notebooks" "tests/unit" "tests/integration" "docs" "data/raw" "data/processed" "data/plots" "agent_logs")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir"
    else
        echo "✗ Missing: $dir"
        exit 1
    fi
done
echo ""

# Check 3: Key files exist
echo "[3/5] Checking key files..."
REQUIRED_FILES=("README.md" "environment.yml" ".gitignore" "agent_logs/init_log.json")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ Missing: $file"
        exit 1
    fi
done
echo ""

# Check 4: Conda/Python environment
echo "[4/5] Checking Python environment..."
if command -v conda &> /dev/null; then
    echo "✓ Conda available: $(conda --version)"
    if conda env list | grep -q qubitpulseopt; then
        echo "✓ qubitpulseopt environment exists"
    else
        echo "⚠ qubitpulseopt environment not created yet"
        echo "  Run: conda env create -f environment.yml"
    fi
elif command -v python3 &> /dev/null; then
    echo "⚠ Conda not found, using system Python: $(python3 --version)"
    echo "  Consider installing Miniconda for reproducible environments"
else
    echo "✗ No Python found"
    exit 1
fi
echo ""

# Check 5: SOW document
echo "[5/5] Checking SOW reference..."
if [ -f "docs/Scope of Work_ Quantum Controls Simulation Project.md" ]; then
    LINE_COUNT=$(wc -l < "docs/Scope of Work_ Quantum Controls Simulation Project.md")
    echo "✓ SOW document copied ($LINE_COUNT lines)"
else
    echo "✗ SOW document missing in docs/"
    exit 1
fi
echo ""

echo "=== Validation Complete ==="
echo "Next steps:"
echo "1. conda env create -f environment.yml"
echo "2. conda activate qubitpulseopt"
echo "3. python -c 'import qutip; qutip.about()'"
