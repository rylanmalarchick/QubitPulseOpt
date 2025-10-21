#!/bin/bash
# Environment activation helper for QubitPulseOpt
# Usage: source scripts/activate_env.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/venv"

echo "=== QubitPulseOpt Environment Activation ==="
echo ""

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo ""
    echo "Please create the environment first:"
    echo "  cd $PROJECT_ROOT"
    echo "  python3 -m venv venv"
    echo "  venv/bin/pip install qutip numpy scipy matplotlib jupyter pytest pytest-cov black flake8 ipykernel"
    echo ""
    return 1 2>/dev/null || exit 1
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✓ Virtual environment activated: $VIRTUAL_ENV"
    echo ""

    # Display package versions
    echo "Installed Packages:"
    python -c "import qutip; print(f'  QuTiP:      {qutip.__version__}')"
    python -c "import numpy; print(f'  NumPy:      {numpy.__version__}')"
    python -c "import scipy; print(f'  SciPy:      {scipy.__version__}')"
    python -c "import matplotlib; print(f'  Matplotlib: {matplotlib.__version__}')"
    python -c "import pytest; print(f'  Pytest:     {pytest.__version__}')"
    echo ""

    echo "Environment ready! You can now:"
    echo "  - Run notebooks: jupyter notebook notebooks/"
    echo "  - Run tests:     pytest tests/ -v"
    echo "  - Deactivate:    deactivate"
    echo ""
else
    echo "❌ Failed to activate virtual environment"
    return 1 2>/dev/null || exit 1
fi
