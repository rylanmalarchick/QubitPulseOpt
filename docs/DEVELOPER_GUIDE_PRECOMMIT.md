# Developer Guide: Pre-commit Hooks

This guide covers the setup and usage of pre-commit hooks for QubitPulseOpt. Pre-commit hooks automatically check your code for quality, style, and compliance issues before you commit, helping maintain code quality and catch issues early.

## Table of Contents
- [What are Pre-commit Hooks?](#what-are-pre-commit-hooks)
- [Quick Start](#quick-start)
- [Available Hooks](#available-hooks)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [CI Integration](#ci-integration)

---

## What are Pre-commit Hooks?

Pre-commit hooks are scripts that run automatically when you execute `git commit`. They check your staged files for issues and can:
- **Auto-fix** formatting and style issues (trailing whitespace, import order, code formatting)
- **Block commits** that violate critical rules (syntax errors, Power of 10 violations)
- **Warn** about potential issues without blocking (complexity warnings, linting suggestions)

This ensures that code quality issues are caught before they enter the repository, making code reviews more focused on logic and design.

---

## Quick Start

### 1. Install Pre-commit Framework

**Using pip (in your activated virtual environment):**
```bash
# Activate your environment first
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install pre-commit
pip install pre-commit
```

**Using conda (if using conda environment):**
```bash
conda activate qubitpulseopt
conda install -c conda-forge pre-commit
```

### 2. Install the Git Hooks

From the project root directory:
```bash
pre-commit install
```

This installs the pre-commit hook into `.git/hooks/pre-commit`. Now the hooks will run automatically on every commit.

### 3. (Optional) Install commit-msg hook

For additional commit message validation:
```bash
pre-commit install --hook-type commit-msg
```

### 4. Verify Installation

```bash
# Check pre-commit version
pre-commit --version

# List installed hooks
pre-commit run --all-files --verbose
```

---

## Available Hooks

Our pre-commit configuration includes the following hooks:

### Standard Checks (Auto-fix)
- **trailing-whitespace**: Removes trailing whitespace from all files
- **end-of-file-fixer**: Ensures files end with a newline
- **mixed-line-ending**: Enforces LF line endings (Unix-style)

### File Validation
- **check-yaml**: Validates YAML syntax (`.yml`, `.yaml` files)
- **check-json**: Validates JSON syntax
- **check-ast**: Validates Python syntax by parsing AST
- **check-merge-conflict**: Detects merge conflict markers
- **check-case-conflict**: Prevents filename case conflicts
- **check-added-large-files**: Blocks files >500KB (prevents accidental binary commits)
- **debug-statements**: Detects `pdb`, `ipdb`, `breakpoint()` calls

### Python Code Quality

#### Black (Auto-fix)
- **Purpose**: Enforces consistent code formatting
- **Style**: 88-character line length, PEP 8 compliant
- **Auto-fix**: Yes - automatically reformats code
- **Configuration**: `.black` compatible with `flake8`

#### isort (Auto-fix)
- **Purpose**: Sorts and organizes Python imports
- **Style**: Black-compatible profile
- **Auto-fix**: Yes - automatically reorganizes imports
- **Sections**: stdlib → third-party → first-party → local

#### Flake8 (Check only)
- **Purpose**: Lints Python code for style and potential errors
- **Max line length**: 88 (Black-compatible)
- **Max complexity**: 10 (McCabe)
- **Ignored rules**:
  - `E203`: Whitespace before ':' (Black compatibility)
  - `W503`: Line break before binary operator (outdated PEP 8)
  - `E501`: Line too long (handled by Black)
- **Plugins**: `flake8-docstrings`, `flake8-bugbear`

### Custom Compliance Checks

#### Power of 10 Compliance (Check only)
- **Purpose**: Enforces NASA/JPL Power of 10 safety-critical coding rules
- **Rules checked**:
  1. Simple control flow (no recursion, <3 nesting levels)
  2. Bounded loops (explicit upper bounds)
  3. No dynamic allocation after init
  4. Function length ≤60 lines
  5. Assertion density ≥2/function
  6. Minimal scope
  7. Check return values
  8. Minimal metaprogramming
  9. Restricted indirection
  10. Zero warnings
- **Mode**: Pre-commit mode only fails on **errors**, not warnings
- **Manual check**: `python scripts/compliance/power_of_10_checker.py src --verbose`

### Notebook Checks (Auto-fix)

#### nbQA-black
- **Purpose**: Formats code cells in Jupyter notebooks
- **Auto-fix**: Yes - reformats notebook code cells

#### nbQA-isort
- **Purpose**: Sorts imports in notebook code cells
- **Auto-fix**: Yes - reorganizes notebook imports

### Markdown Linting (Auto-fix)

#### markdownlint
- **Purpose**: Lints Markdown files for consistency
- **Auto-fix**: Yes - fixes many style issues automatically
- **Disabled rules**:
  - `MD013`: Line length (we allow long lines)
  - `MD033`: Inline HTML (used for badges)
  - `MD041`: First line must be H1 (not always needed)

---

## Configuration

### `.pre-commit-config.yaml`

The main configuration file. Key sections:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        args: ['--line-length=88', '--target-version=py39']
```

### Updating Hook Versions

Pre-commit hooks reference specific versions (tags) of external repositories. To update:

```bash
# Update all hooks to latest versions
pre-commit autoupdate

# Update specific hook
pre-commit autoupdate --repo https://github.com/psf/black
```

### Customizing Behavior

To skip specific hooks for a commit:
```bash
# Skip all hooks (not recommended)
git commit --no-verify

# Skip specific hooks
SKIP=flake8,power-of-10-compliance git commit -m "WIP: refactoring"
```

To disable a hook permanently, comment it out in `.pre-commit-config.yaml`:
```yaml
# - id: some-hook-to-disable
```

---

## Usage

### Normal Workflow

1. **Make your changes** as usual
2. **Stage files**: `git add <files>`
3. **Commit**: `git commit -m "Your message"`
4. **Hooks run automatically**:
   - Auto-fixable issues are corrected
   - Files are re-staged automatically
   - Commit proceeds if all checks pass
   - Commit is blocked if critical issues remain

### Example Session

```bash
$ git add src/optimization/gates.py
$ git commit -m "Add gate fidelity optimization"

Trim trailing whitespace...........................................Passed
Fix end of files...............................................Passed
Check YAML.....................................................Passed
Check JSON.....................................................Passed
Check for large files..........................................Passed
Check for merge conflicts......................................Passed
Format code with Black.........................................Failed
- hook id: black
- files were modified by this hook

reformatted src/optimization/gates.py
1 file reformatted.

Sort imports with isort........................................Passed
Lint with Flake8...............................................Passed
Power of 10 Compliance Check...................................Passed

# Files were auto-fixed, try commit again
$ git commit -m "Add gate fidelity optimization"

# ... all checks pass ...
[main abc1234] Add gate fidelity optimization
 1 file changed, 50 insertions(+)
```

### Manual Hook Execution

Run hooks manually without committing:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run all hooks on staged files only
pre-commit run

# Run specific hook on all files
pre-commit run black --all-files

# Run specific hook on specific files
pre-commit run flake8 --files src/optimization/*.py

# Show verbose output
pre-commit run --all-files --verbose
```

### Bypassing Hooks (When Necessary)

Sometimes you need to commit code that doesn't pass all checks (e.g., work-in-progress):

```bash
# Bypass all hooks (use sparingly!)
git commit --no-verify -m "WIP: work in progress"

# Bypass specific hooks
SKIP=flake8 git commit -m "WIP: refactoring in progress"

# Bypass Power of 10 compliance only
SKIP=power-of-10-compliance git commit -m "Temporary: complex function to be refactored"
```

**⚠️ Warning**: Bypassed hooks still run in CI, so the PR may fail checks later.

---

## Troubleshooting

### Installation Issues

**Problem**: `pre-commit: command not found`
```bash
# Solution: Install pre-commit
pip install pre-commit
# OR
conda install -c conda-forge pre-commit
```

**Problem**: Hooks not running on commit
```bash
# Solution: Reinstall hooks
pre-commit uninstall
pre-commit install
```

**Problem**: Wrong Python interpreter used
```bash
# Solution: Install pre-commit in the same environment as your project
source venv/bin/activate  # Activate your project's venv first
pip install pre-commit
pre-commit install
```

### Hook Failures

**Problem**: Black reformats code differently than expected
```bash
# Solution: Check Black configuration in .pre-commit-config.yaml
# Ensure line-length matches flake8 configuration (88)
```

**Problem**: Flake8 and Black conflict on line length
```bash
# Solution: Both should use 88-character line length
# Flake8 should ignore E203, W503 for Black compatibility
```

**Problem**: Power of 10 checker fails with false positives
```bash
# Solution 1: Check verbose output for specific rule violations
python scripts/compliance/power_of_10_checker.py src/your_file.py --verbose

# Solution 2: If truly a false positive, document with comment and skip hook for that commit
SKIP=power-of-10-compliance git commit -m "..."
```

**Problem**: markdownlint fails on legitimate Markdown
```bash
# Solution: Disable specific rules in .pre-commit-config.yaml
# Add to markdownlint args: ['--disable', 'MD999']
```

### Performance Issues

**Problem**: Pre-commit hooks take too long
```bash
# Solution 1: Run only on changed files (default behavior)
# Don't use --all-files unless necessary

# Solution 2: Skip slow hooks for WIP commits
SKIP=power-of-10-compliance,flake8 git commit -m "WIP"

# Solution 3: Use pre-commit.ci (cloud service) instead of local hooks
# See: https://pre-commit.ci
```

### Cleaning Up

```bash
# Uninstall hooks
pre-commit uninstall

# Clean pre-commit cache
pre-commit clean

# Remove all pre-commit environments
rm -rf ~/.cache/pre-commit/
```

---

## CI Integration

### GitHub Actions

Pre-commit hooks are also enforced in CI via GitHub Actions workflows:

#### Fast CI (`.github/workflows/fast-tests.yml`)
- Runs on every push
- Executes: `pre-commit run --all-files`
- Fast feedback loop (< 5 minutes)

#### Full CI (`.github/workflows/full-tests.yml`)
- Runs on pull requests
- Includes full test suite + compliance checks
- Comprehensive validation

#### Compliance CI (`.github/workflows/compliance.yml`)
- Dedicated compliance checking workflow
- Runs Power of 10 checker with full reporting
- Generates compliance reports as artifacts

### Pre-commit.ci (Optional Cloud Service)

We've configured support for [pre-commit.ci](https://pre-commit.ci), a free cloud service that:
- Runs pre-commit hooks on every PR
- Auto-fixes issues and pushes fixes to PR branch
- Updates hook versions quarterly

To enable:
1. Visit https://pre-commit.ci
2. Sign in with GitHub
3. Enable for the `QubitPulseOpt` repository

**Note**: Custom hooks (like Power of 10 checker) are skipped in pre-commit.ci and only run in GitHub Actions.

---

## Best Practices

### Do's ✅
- **Install pre-commit hooks immediately** after cloning the repo
- **Let auto-fix hooks do their job** - don't manually reformat after Black/isort run
- **Run `pre-commit run --all-files`** after pulling main branch changes
- **Update hooks quarterly**: `pre-commit autoupdate`
- **Use verbose mode** to understand failures: `pre-commit run --verbose --all-files`
- **Commit the auto-fixes** from Black/isort without modification

### Don'ts ❌
- **Don't use `--no-verify` habitually** - bypassing hooks defeats their purpose
- **Don't fight the formatter** - if Black reformats your code, accept it
- **Don't commit large files** - the 500KB limit is there for a reason
- **Don't disable all hooks** - at minimum, keep syntax and safety checks
- **Don't ignore Power of 10 errors** - they catch real safety/maintainability issues

### Workflow Tips

**Starting a new feature branch:**
```bash
git checkout -b feature/my-feature
# Run hooks on all files to ensure clean baseline
pre-commit run --all-files
```

**Before opening a PR:**
```bash
# Ensure all files pass all hooks
pre-commit run --all-files

# Run full test suite
pytest tests/ -v

# Check compliance report
python scripts/compliance/power_of_10_checker.py src --verbose
```

**Dealing with legacy code:**
```bash
# Refactor gradually - use SKIP for interim commits if needed
SKIP=power-of-10-compliance git commit -m "Refactor step 1 of 3"

# But ensure final commit passes all checks
git commit -m "Refactor complete - all compliance checks pass"
```

---

## Additional Resources

- **Pre-commit Documentation**: https://pre-commit.com
- **Black Documentation**: https://black.readthedocs.io
- **Flake8 Documentation**: https://flake8.pycqa.org
- **isort Documentation**: https://pycqa.github.io/isort/
- **Power of 10 Rules**: See `docs/Scope of Work*.md` Section on coding standards
- **Project Testing Guide**: `tests/README_TESTING.md`

---

## Quick Reference Card

### Common Commands

| Command | Description |
|---------|-------------|
| `pre-commit install` | Install hooks (run once after clone) |
| `pre-commit run` | Run on staged files |
| `pre-commit run --all-files` | Run on all files in repo |
| `pre-commit run black --all-files` | Run specific hook |
| `pre-commit autoupdate` | Update hook versions |
| `pre-commit clean` | Clear cache |
| `SKIP=hook1,hook2 git commit` | Skip specific hooks |
| `git commit --no-verify` | Skip all hooks (use sparingly!) |

### Hook Status Indicators

| Status | Meaning |
|--------|---------|
| `Passed` ✅ | Hook passed, no issues found |
| `Failed` ❌ | Hook found issues, commit blocked |
| `Skipped` ⏭️ | Hook skipped (no matching files) |
| `Fixed` 🔧 | Hook auto-fixed issues (re-run commit) |

---

**Last Updated**: 2024
**Maintainer**: QubitPulseOpt Development Team