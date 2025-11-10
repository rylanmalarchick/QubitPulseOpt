# Pre-commit Hooks - Quick Reference Card

**One-page guide for QubitPulseOpt pre-commit hooks**

---

## 🚀 First-Time Setup

```bash
# Option 1: Automated (Recommended)
./scripts/setup_dev_env.sh

# Option 2: Manual
pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files  # Initial run
```

---

## 📋 Common Commands

| Command | Description |
|---------|-------------|
| `pre-commit install` | Install hooks (one-time setup) |
| `pre-commit run` | Run on staged files |
| `pre-commit run --all-files` | Run on entire repository |
| `pre-commit run black --all-files` | Run specific hook |
| `pre-commit autoupdate` | Update hook versions |
| `pre-commit clean` | Clear cache |
| `pre-commit uninstall` | Remove hooks |

---

## 🔧 Hooks Installed

### Auto-Fix Hooks (modify files automatically)
- ✅ **black** - Code formatter (88 chars)
- ✅ **isort** - Import sorter
- ✅ **trailing-whitespace** - Remove trailing spaces
- ✅ **end-of-file-fixer** - Add final newline
- ✅ **mixed-line-ending** - Fix line endings (LF)
- ✅ **markdownlint** - Markdown formatter

### Check-Only Hooks (report issues, don't modify)
- ⚠️ **flake8** - Python linter
- ⚠️ **power-of-10-compliance** - Safety standards
- ⚠️ **check-yaml** - YAML syntax
- ⚠️ **check-json** - JSON syntax
- ⚠️ **check-ast** - Python syntax
- ⚠️ **debug-statements** - Catch `pdb`, `breakpoint()`
- ⚠️ **check-added-large-files** - Block files >500KB

---

## 🎯 Typical Workflow

```bash
# 1. Make changes to code
vim src/optimization/gates.py

# 2. Stage changes
git add src/optimization/gates.py

# 3. Commit (hooks run automatically)
git commit -m "Add fidelity optimization"

# If hooks auto-fix files:
# - Review changes: git diff
# - Commit again: git commit -m "Add fidelity optimization"
```

---

## ⚡ Bypass Hooks (Use Sparingly!)

```bash
# Skip ALL hooks (not recommended)
git commit --no-verify -m "WIP: work in progress"

# Skip specific hooks
SKIP=flake8,power-of-10-compliance git commit -m "WIP"

# Skip just compliance
SKIP=power-of-10-compliance git commit -m "Refactoring in progress"
```

**⚠️ Warning:** Bypassed hooks still run in CI, so PRs may fail later!

---

## 🐛 Troubleshooting

### Hook not running?
```bash
pre-commit uninstall
pre-commit install
```

### Hooks taking too long?
```bash
# Don't run on all files unless needed
pre-commit run  # Only staged files

# Skip slow hooks for WIP
SKIP=power-of-10-compliance git commit -m "WIP"
```

### Conflict between Black and Flake8?
```bash
# Both should use 88-char line length
# Flake8 ignores E203, W503 for Black compatibility
# If still failing, check .flake8 config
```

### Power of 10 false positive?
```bash
# Check specific file in verbose mode
python scripts/compliance/power_of_10_checker.py src/yourfile.py --verbose

# Skip for this commit, fix later
SKIP=power-of-10-compliance git commit -m "..."
```

### Hook environments corrupted?
```bash
pre-commit clean
pre-commit install --install-hooks
```

---

## 📊 Hook Status Indicators

| Symbol | Meaning |
|--------|---------|
| ✅ **Passed** | No issues found |
| ❌ **Failed** | Issues found, commit blocked |
| 🔧 **Fixed** | Auto-fixed issues, re-commit |
| ⏭️ **Skipped** | No matching files |

---

## 🎨 Code Style Standards

### Black (Formatter)
- Line length: **88 characters**
- Target: **Python 3.9+**
- Style: **PEP 8 compliant**

### Flake8 (Linter)
- Max line length: **88**
- Max complexity: **10**
- Ignored: **E203, W503, E501** (Black compat)
- Plugins: **docstrings, bugbear**

### isort (Import Sorter)
- Profile: **Black-compatible**
- Order: **stdlib → third-party → first-party → local**

### Power of 10 (Compliance)
1. No recursion, nesting <3 levels
2. All loops bounded
3. Functions ≤60 lines
4. Assertion density ≥2/function
5. Validate inputs/outputs

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| `docs/DEVELOPER_GUIDE_PRECOMMIT.md` | Comprehensive guide (495 lines) |
| `CONTRIBUTING.md` | Contribution guidelines |
| `.pre-commit-config.yaml` | Hook configuration |
| `pyproject.toml` | Tool settings |
| `.flake8` | Flake8 config |

---

## 🔄 Before Opening a PR

```bash
# 1. Run all hooks
pre-commit run --all-files

# 2. Run tests
pytest tests/ -v --cov=src

# 3. Check compliance
python scripts/compliance/power_of_10_checker.py src --verbose

# 4. Review changes
git diff main..HEAD

# 5. Push and create PR
git push origin feature/your-branch
```

---

## 💡 Pro Tips

✅ **DO:**
- Let auto-fix hooks do their job
- Run `--all-files` after pulling main
- Update hooks quarterly: `pre-commit autoupdate`
- Review auto-fixes before re-committing

❌ **DON'T:**
- Use `--no-verify` habitually
- Fight the formatter (accept Black's style)
- Commit large binary files
- Disable hooks permanently

---

## 🆘 Getting Help

- **Quick help**: This file
- **Detailed guide**: `docs/DEVELOPER_GUIDE_PRECOMMIT.md`
- **Issues**: Open GitHub issue
- **Testing**: `tests/README_TESTING.md`

---

**Last Updated:** 2024  
**Quick Links:**
- Pre-commit docs: https://pre-commit.com
- Black docs: https://black.readthedocs.io
- Flake8 docs: https://flake8.pycqa.org

---

**Need more detail?** See `docs/DEVELOPER_GUIDE_PRECOMMIT.md`
