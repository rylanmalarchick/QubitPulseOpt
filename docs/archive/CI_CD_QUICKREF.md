# CI/CD Pipeline Quick Reference Guide

**QubitPulseOpt Project**  
**Last Updated:** 2024

---

## 📋 Overview

The QubitPulseOpt project uses GitHub Actions for continuous integration and deployment. Every push and pull request triggers automated quality checks to ensure code quality, compliance, and documentation.

---

## 🚀 Active Workflows

### 1. **Tests** (`tests.yml`)
**Triggers:** Push to main/develop, Pull Requests  
**Duration:** ~5-10 minutes

**What it does:**
- Runs full test suite on Python 3.9, 3.10, 3.11, 3.12
- Generates code coverage reports
- Uploads coverage to Codecov
- Runs slow tests nightly or on-demand

**How to pass:**
- All tests must pass on all Python versions
- No new test failures introduced

**Manual slow test run:**
```bash
git commit -m "your message [run-slow]"
```

---

### 2. **Compliance** (`compliance.yml`)
**Triggers:** Push to main/develop, Pull Requests  
**Duration:** ~2-3 minutes

**What it does:**
- Runs Power-of-10 compliance checker
- Enforces 97% minimum compliance score
- Zero-tolerance for Rule 4 violations (recursion)
- Posts detailed compliance report on PRs

**How to pass:**
- Maintain compliance score ≥ 97%
- No recursion (Rule 4 violations)
- Address critical violations before pushing

**Check locally:**
```bash
python scripts/compliance/power_of_10_checker.py src --verbose
```

---

### 3. **Linting** (`lint.yml`)
**Triggers:** Push to main/develop, Pull Requests  
**Duration:** ~3-4 minutes

**What it does:**
- Checks code formatting with Black
- Validates import ordering with isort
- Runs flake8 linting
- Optional mypy type checking
- Security scanning (Bandit, Safety)
- Docstring coverage analysis

**How to pass:**
- Code must be formatted with Black
- Imports must be sorted with isort
- No critical flake8 errors

**Fix locally:**
```bash
# Auto-fix formatting and imports
black src/ tests/
isort src/ tests/

# Check for issues
flake8 src/ tests/
```

---

### 4. **Documentation** (`docs.yml`)
**Triggers:** Push to main/develop, Pull Requests  
**Duration:** ~3-5 minutes

**What it does:**
- Builds Sphinx documentation
- Validates markdown files
- Checks Jupyter notebooks
- Deploys to GitHub Pages (main branch only)

**How to pass:**
- Documentation builds without errors
- No broken links or syntax errors

**Build locally:**
```bash
cd docs
sphinx-build -b html . _build/html
```

---

### 5. **Pre-commit** (`pre-commit.yml`)
**Triggers:** Push to main/develop, Pull Requests  
**Duration:** ~2-3 minutes

**What it does:**
- Validates pre-commit hooks configuration
- Runs all pre-commit checks in CI
- Ensures consistency with local checks

**How to pass:**
- Pre-commit hooks must pass
- Same checks as local pre-commit

**Setup locally:**
```bash
./scripts/setup_dev_env.sh
# or manually:
pip install pre-commit
pre-commit install
```

---

## ✅ Quality Gates

Your code must pass ALL of these to be merged:

1. ✅ **Tests pass** on Python 3.9, 3.10, 3.11, 3.12
2. ✅ **Compliance score ≥ 97%**
3. ✅ **No Rule 4 violations** (recursion forbidden)
4. ✅ **Black formatting** compliant
5. ✅ **isort import ordering** compliant
6. ✅ **No critical flake8 errors**
7. ✅ **Documentation builds** successfully

---

## 🔧 Common Issues & Fixes

### ❌ Tests Failing

**Problem:** Test failures on specific Python version

**Fix:**
```bash
# Test locally with specific Python version
python3.10 -m pytest tests/ -v

# Or use tox/nox for multi-version testing
```

---

### ❌ Compliance Score Too Low

**Problem:** Compliance score < 97%

**Fix:**
```bash
# Check compliance locally
python scripts/compliance/power_of_10_checker.py src --verbose

# Focus on:
# - Rule 1: Simplify control flow
# - Rule 2: Add loop bounds
# - Rule 4: Remove recursion (CRITICAL)
# - Rule 5: Add assertions (min 2 per function)
```

---

### ❌ Black Formatting Issues

**Problem:** "Black formatting issues found"

**Fix:**
```bash
# Auto-fix all formatting
black src/ tests/

# Check what would change (dry run)
black --check --diff src/ tests/
```

---

### ❌ Import Ordering Issues

**Problem:** "Import ordering issues found"

**Fix:**
```bash
# Auto-fix import ordering
isort src/ tests/

# Check what would change
isort --check-only --diff src/ tests/
```

---

### ❌ Flake8 Errors

**Problem:** Linting errors detected

**Fix:**
```bash
# See what's wrong
flake8 src/ tests/

# Common issues:
# - Line too long (max 100 chars)
# - Unused imports
# - Undefined names
# - Complexity too high (max 15)
```

---

### ❌ Documentation Build Fails

**Problem:** Sphinx build errors

**Fix:**
```bash
# Build docs locally to see errors
cd docs
sphinx-build -b html . _build/html

# Common issues:
# - Malformed docstrings
# - Missing references
# - Invalid RST syntax
```

---

## 🎯 Best Practices

### Before Pushing

1. **Run pre-commit hooks:**
   ```bash
   pre-commit run --all-files
   ```

2. **Run tests locally:**
   ```bash
   pytest tests/ -v
   ```

3. **Check compliance:**
   ```bash
   python scripts/compliance/power_of_10_checker.py src --verbose
   ```

4. **Format code:**
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

### During Development

- Write tests for new features
- Add docstrings to new functions
- Keep functions under 60 lines (Rule 4)
- Add at least 2 assertions per function (Rule 5)
- Avoid recursion (Rule 4 - critical)

### When Creating PRs

- Check CI status before requesting review
- Address all compliance warnings
- Review coverage reports
- Update documentation if needed

---

## 📊 Monitoring & Reports

### Where to Find Reports

**Coverage Reports:**
- Codecov: Linked in PR checks
- HTML Report: Download from workflow artifacts

**Compliance Reports:**
- PR Comment: Automatic on all PRs
- JSON Report: Download from workflow artifacts

**Lint Reports:**
- Workflow summary: GitHub Actions UI
- Detailed reports: Download from artifacts

**Security Reports:**
- Bandit results: Workflow artifacts
- Safety results: Workflow artifacts

### Artifact Retention

- Coverage reports: 30 days
- Compliance reports: 90 days
- Security reports: 90 days
- Documentation: 90 days

---

## 🚨 Critical Violations

These will **immediately fail** your build:

### Rule 4 Violations (Recursion)
```python
# ❌ BAD - Will fail CI
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # Recursion forbidden!

# ✅ GOOD - Use iteration
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
```

### Critical Flake8 Errors
- F821: Undefined name
- E999: Syntax error
- F631: Assertion test is a tuple (always true)

### Test Failures
- Any test failing on any Python version
- Import errors
- Assertion errors

---

## 🔄 Workflow Status

### Check Status

**In GitHub UI:**
- Green ✅: All checks passed
- Yellow 🟡: Checks in progress
- Red ❌: Checks failed

**In Terminal:**
```bash
# Check latest workflow runs
gh run list

# View specific run
gh run view <run-id>

# Watch live
gh run watch
```

### Re-running Failed Checks

1. Fix the issue locally
2. Push new commit
3. CI runs automatically

Or re-run from GitHub UI:
- Go to Actions tab
- Click failed workflow
- Click "Re-run jobs"

---

## 📚 Additional Resources

### Documentation
- **Pre-commit Guide:** `docs/DEVELOPER_GUIDE_PRECOMMIT.md`
- **Contributing Guide:** `CONTRIBUTING.md`
- **Pre-commit Quick Ref:** `PRECOMMIT_QUICKREF.md`
- **Full CI/CD Summary:** `TASK_4_CICD_COMPLETE.md`

### Tools
- **Black:** https://black.readthedocs.io/
- **isort:** https://pycqa.github.io/isort/
- **flake8:** https://flake8.pycqa.org/
- **pytest:** https://docs.pytest.org/
- **Sphinx:** https://www.sphinx-doc.org/

### Power of 10 Rules
See: `docs/POWER_OF_10_RULES.md` (or NASA's JPL guidelines)

---

## 💡 Tips & Tricks

### Speed Up Local Checks

```bash
# Run only on changed files
pre-commit run

# Run specific hook
pre-commit run black

# Skip slow hooks during development
SKIP=compliance-check git commit -m "WIP"
```

### Parallel Testing

```bash
# Run tests in parallel (faster)
pytest tests/ -n auto

# Run only fast tests during development
pytest tests/ -m "not slow"
```

### Auto-fix Everything

```bash
# One command to fix common issues
black src/ tests/ && isort src/ tests/ && pre-commit run --all-files
```

### Bypass Hooks (Emergency Only)

```bash
# Skip pre-commit hooks (not recommended)
git commit --no-verify -m "Emergency fix"

# Note: CI will still run all checks!
```

---

## 🆘 Getting Help

### Workflow Failed - What Now?

1. **Read the error message** in GitHub Actions
2. **Download artifacts** for detailed reports
3. **Run checks locally** to reproduce
4. **Fix the issue** and push again
5. **Ask for help** if stuck (include error logs)

### Common Questions

**Q: Can I skip a specific check?**  
A: No. All quality gates are required. Fix the issue instead.

**Q: My test is flaky. Can I disable it?**  
A: Mark it with `@pytest.mark.flaky(reruns=3)` or fix the flakiness.

**Q: Compliance check is too strict!**  
A: Compliance ensures code quality. If you have a valid reason, discuss with the team.

**Q: How do I add a new workflow?**  
A: Create YAML in `.github/workflows/` and follow existing patterns.

---

## 📈 Continuous Improvement

The CI/CD pipeline is designed to:
- ✅ Catch issues early
- ✅ Maintain code quality
- ✅ Enable confident collaboration
- ✅ Automate repetitive tasks

**Remember:** These checks exist to help you write better code, not to slow you down. Embrace them! 🚀

---

**Last Updated:** 2024  
**Maintained By:** QubitPulseOpt Team  
**Questions?** Open an issue or ask in team chat