# Pre-commit Hooks Migration Guide

**For Existing QubitPulseOpt Developers**

This guide helps you migrate your existing development environment to use the new pre-commit hook infrastructure.

---

## Who Should Read This?

If you:
- Already have QubitPulseOpt cloned and are working on it
- Have made commits without pre-commit hooks
- Need to update your local environment to match the new standards

Then this guide is for you!

---

## Why Pre-commit Hooks?

Before pre-commit hooks:
- ❌ Code style inconsistencies slip through
- ❌ CI catches issues 15+ minutes after commit
- ❌ Manual formatting is time-consuming
- ❌ Code reviews focus on style instead of logic

After pre-commit hooks:
- ✅ Automatic code formatting on every commit
- ✅ Issues caught in <10 seconds, before push
- ✅ Zero manual formatting needed
- ✅ Code reviews focus on what matters

---

## Migration Steps

### Step 1: Update Your Local Repository

```bash
# Navigate to your QubitPulseOpt directory
cd /path/to/QubitPulseOpt

# Fetch latest changes
git fetch origin main

# If on a feature branch, rebase or merge
git checkout main
git pull origin main

# Update your feature branch (if applicable)
git checkout your-feature-branch
git rebase main  # or: git merge main
```

### Step 2: Install Development Dependencies

**If using venv (most common):**
```bash
# Activate your existing venv
source venv/bin/activate

# Install new development dependencies
pip install -r requirements-dev.txt

# Verify installation
pre-commit --version  # Should show v3.5.0 or higher
```

**If using conda:**
```bash
# Activate your conda environment
conda activate qubitpulseopt

# Install pre-commit
conda install -c conda-forge pre-commit

# Install other dev tools
pip install -r requirements-dev.txt
```

### Step 3: Install Git Hooks

```bash
# Install pre-commit git hooks
pre-commit install

# (Optional) Install commit-msg hook
pre-commit install --hook-type commit-msg

# Verify installation
ls -la .git/hooks/pre-commit  # Should exist
```

### Step 4: Run Pre-commit on Existing Code

```bash
# Run all hooks on all files (this will take 2-5 minutes)
pre-commit run --all-files

# If files are modified (Black/isort auto-fix):
# Review the changes
git diff

# If changes look good, commit them
git add -u
git commit -m "Apply pre-commit auto-formatting"
```

**Expected Output:**
```
Trim trailing whitespace...........................................Passed
Fix end of files...............................................Passed
Check YAML.....................................................Passed
...
Format code with Black.........................................Failed
- hook id: black
- files were modified by this hook

reformatted src/optimization/gates.py
...

# This is normal! Black reformatted your code.
# Review changes with: git diff
# Then commit: git commit -am "Apply pre-commit formatting"
```

### Step 5: Configure Your Workflow

**Optional but recommended:**

Create a shell alias for convenience:
```bash
# Add to your ~/.bashrc or ~/.zshrc
alias pc='pre-commit run --all-files'
alias pcf='pre-commit run --files'

# Reload shell
source ~/.bashrc  # or ~/.zshrc
```

---

## Dealing with Existing Work-in-Progress

### Scenario A: Clean Feature Branch

If your feature branch has clean, ready-to-merge code:

```bash
# Run pre-commit and fix all issues
pre-commit run --all-files

# Commit any auto-fixes
git add -u
git commit -m "Apply pre-commit formatting to feature"

# Continue development normally
```

### Scenario B: Messy Work-in-Progress

If you have WIP commits that don't pass checks:

**Option 1: Fix Issues (Recommended)**
```bash
# Run pre-commit
pre-commit run --all-files

# Fix any issues it finds
# Commit fixes
git add -u
git commit -m "Fix pre-commit issues"
```

**Option 2: Bypass for Now, Fix Later**
```bash
# Make WIP commits with bypass
git commit --no-verify -m "WIP: work in progress"

# When ready for PR, fix all issues
pre-commit run --all-files
git add -u
git commit -m "Fix code quality issues before PR"
```

**Option 3: Interactive Rebase (Advanced)**
```bash
# Rewrite history to apply formatting retroactively
git rebase -i main

# For each commit:
# - Apply pre-commit formatting
# - Amend the commit
# This creates a clean history
```

### Scenario C: Merge Conflicts

If you get merge conflicts after updating from main:

```bash
# Resolve conflicts as normal
vim conflicted_file.py

# Stage resolved files
git add conflicted_file.py

# Commit (pre-commit will run automatically)
git commit -m "Merge main into feature branch"

# If pre-commit modifies files:
git add -u
git commit --amend --no-edit
```

---

## Common Migration Issues

### Issue 1: Black Reformats Everything

**Problem:** Black reformats many files on first run.

**Solution:** This is expected! Black enforces consistent style.
```bash
# Review changes
git diff

# If changes are reasonable (they usually are), accept them
git add -u
git commit -m "Apply Black formatting"

# Don't fight Black - its style is deliberate and consistent
```

### Issue 2: Flake8 Finds Many Issues

**Problem:** Flake8 reports linting errors.

**Solution:** Fix them incrementally.
```bash
# See all errors
pre-commit run flake8 --all-files

# Fix file by file
vim src/problematic_file.py  # Fix issues
pre-commit run flake8 --files src/problematic_file.py

# Or skip for WIP commits
SKIP=flake8 git commit -m "WIP: will fix linting later"
```

### Issue 3: Power of 10 Compliance Failures

**Problem:** Power of 10 checker reports violations.

**Solution:** These need to be fixed before merging.
```bash
# See detailed report
python scripts/compliance/power_of_10_checker.py src --verbose

# Fix violations in order of severity:
# 1. Errors (must fix)
# 2. Warnings (should fix)

# For WIP, bypass temporarily
SKIP=power-of-10-compliance git commit -m "WIP"
```

### Issue 4: Hooks Are Slow

**Problem:** Pre-commit takes too long.

**Solution:** 
```bash
# Only run on changed files (default behavior)
git commit  # Only checks staged files

# Don't run --all-files unless needed
pre-commit run  # Fast

# Skip slow hooks for quick WIP commits
SKIP=power-of-10-compliance git commit -m "WIP"
```

### Issue 5: Hook Installation Failed

**Problem:** `pre-commit install` fails.

**Solution:**
```bash
# Ensure pre-commit is installed in the right environment
which pre-commit  # Should be in your venv/conda env

# If not, install it
pip install pre-commit  # or: conda install -c conda-forge pre-commit

# Clean and reinstall
pre-commit uninstall
pre-commit clean
pre-commit install
```

---

## Updating Existing Pull Requests

If you have open PRs created before pre-commit hooks:

### Step 1: Update Your Branch

```bash
# Checkout PR branch
git checkout your-pr-branch

# Merge or rebase main
git merge main  # or: git rebase main
```

### Step 2: Install and Run Hooks

```bash
# Install hooks (if not done already)
pre-commit install

# Run on all files
pre-commit run --all-files

# Commit any changes
git add -u
git commit -m "Apply pre-commit formatting"
```

### Step 3: Push Updates

```bash
# Push to your PR
git push origin your-pr-branch

# If you rebased, force push
git push --force-with-lease origin your-pr-branch
```

The PR will now pass CI pre-commit checks!

---

## Best Practices Going Forward

### DO ✅

1. **Run pre-commit before pushing**
   ```bash
   pre-commit run --all-files
   git push
   ```

2. **Let auto-fix hooks work**
   - Don't manually reformat after Black runs
   - Accept Black's style choices

3. **Fix issues early**
   - Address linting errors as you code
   - Don't accumulate technical debt

4. **Update hooks regularly**
   ```bash
   pre-commit autoupdate  # Every 3 months
   ```

5. **Use bypass sparingly**
   ```bash
   # Only for legitimate WIP
   git commit --no-verify -m "WIP: checkpoint"
   ```

### DON'T ❌

1. **Don't bypass habitually**
   - Hooks are there to help you
   - Bypassed hooks still fail in CI

2. **Don't fight the formatter**
   - Black's choices are consistent
   - Accept the style, move on

3. **Don't ignore Power of 10 errors**
   - They catch real issues
   - Fix them before merging

4. **Don't commit large files**
   - 500KB limit is deliberate
   - Use Git LFS if needed

5. **Don't disable hooks permanently**
   - Keep at least syntax/safety checks
   - Talk to team if a hook is problematic

---

## Team Coordination

### For Teams Migrating Together

1. **Announce migration date**
   - Give team 1 week notice
   - Share this guide

2. **Coordinate large formatting changes**
   - One person runs Black on entire codebase
   - Merge that PR first
   - Then everyone rebases their branches

3. **Update CI/CD**
   - Ensure CI runs same checks
   - Update PR requirements

4. **Hold a Q&A session**
   - Answer questions
   - Demonstrate workflow
   - Share tips

### Communication Template

```
Team: We're adopting pre-commit hooks!

What: Automatic code quality checks on every commit
When: [Date] - Please migrate by [Date + 1 week]
How: Follow docs/PRECOMMIT_MIGRATION_GUIDE.md

Key commands:
  pip install -r requirements-dev.txt
  pre-commit install
  pre-commit run --all-files

Questions? Ask in #dev-channel or see the guide.
```

---

## Rollback Procedure (If Needed)

If you need to temporarily disable hooks:

```bash
# Uninstall hooks
pre-commit uninstall

# Work normally without hooks
git commit -m "Message"

# Re-enable later
pre-commit install
```

**Note:** This is only for your local environment. CI will still run checks.

---

## Verification Checklist

After migration, verify everything works:

- [ ] `pre-commit --version` shows v3.5.0+
- [ ] `pre-commit run --all-files` completes successfully
- [ ] `.git/hooks/pre-commit` file exists
- [ ] Test commit triggers hooks automatically
- [ ] Hooks complete in <10 seconds for small changes
- [ ] Can bypass with `--no-verify` if needed
- [ ] CI pre-commit workflow passes

---

## Getting Help

### Documentation
- **Quick Reference**: `PRECOMMIT_QUICKREF.md`
- **Comprehensive Guide**: `docs/DEVELOPER_GUIDE_PRECOMMIT.md`
- **Contributing Guide**: `CONTRIBUTING.md`

### Troubleshooting
- **Common Issues**: See [Common Migration Issues](#common-migration-issues) above
- **Hook Errors**: Run with `--verbose` for details
- **Team Chat**: Ask in developer Slack/Discord

### Support
- **GitHub Issues**: Report bugs/problems
- **Team Lead**: Contact for urgent issues
- **This Guide**: Re-read relevant sections

---

## Timeline

### Week 1: Migration Period
- [ ] All developers install hooks
- [ ] Run `pre-commit run --all-files` on main branch
- [ ] Commit formatting changes

### Week 2: Enforcement
- [ ] CI enforces pre-commit checks
- [ ] All PRs must pass pre-commit
- [ ] Team is comfortable with workflow

### Ongoing
- [ ] Quarterly hook updates
- [ ] Monitor for issues
- [ ] Refine configuration as needed

---

## Success Metrics

After successful migration, you should see:

1. **Faster Code Reviews**
   - No style comments
   - Focus on logic and design

2. **Fewer CI Failures**
   - Issues caught before push
   - Green builds more often

3. **Consistent Code Style**
   - Entire codebase follows Black
   - Imports organized uniformly

4. **Better Code Quality**
   - Power of 10 compliance
   - Fewer bugs slip through

---

## FAQ

**Q: Can I use a different code formatter?**
A: No, the team uses Black for consistency. All code must pass Black formatting.

**Q: What if a hook has a false positive?**
A: Use `SKIP=hookname` for that commit, then report the issue. We'll fix the hook or add an exception.

**Q: Do hooks run on every commit?**
A: Yes, on every commit to any branch. They only check staged files, so they're fast.

**Q: Can I run hooks manually without committing?**
A: Yes! `pre-commit run` runs hooks on staged files without committing.

**Q: What if I forget to install hooks?**
A: CI will catch issues, but you'll waste time. Install hooks to catch issues early.

**Q: Are hooks required?**
A: Yes. CI enforces the same checks. Hooks just give you faster feedback.

---

## Conclusion

Pre-commit hooks are now part of the QubitPulseOpt development workflow. The one-time migration effort pays off with:

- Consistent code quality
- Faster development cycles
- Better code reviews
- Fewer bugs

**Next Steps:**
1. Complete migration steps above
2. Test workflow with a small commit
3. Share feedback with the team
4. Help others migrate

**Welcome to the new workflow!** 🚀

---

**Last Updated:** 2024  
**Maintained By:** QubitPulseOpt Development Team  
**Questions?** Open a GitHub issue or ask in #dev-channel