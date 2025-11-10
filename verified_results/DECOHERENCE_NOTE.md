# Note on Decoherence Evaluation

The decoherence evaluation in the verification script has **unit conversion issues** and produces unreliable results.

## Problem

GRAPE optimization uses dimensionless "natural units" where energies and times are scaled arbitrarily. Converting these to real-world seconds for decoherence evaluation requires knowing the actual energy scales (qubit frequency, Rabi rates, etc.), which are not uniquely determined from the dimensionless optimization.

## Status

- **Closed-system results (99.14% fidelity, 77× error reduction)**: ✅ Valid and reproducible
- **Decoherence evaluation**: ❌ Has unit conversion bugs, results not reliable
- **Future work**: Implement proper open-system GRAPE with decoherence during optimization

## What This Means

The preprint correctly states that optimization was performed in the closed quantum system approximation. The 99.14% fidelity represents performance without decoherence. Real hardware performance would be lower due to T1/T2 effects, but quantifying this requires either:
1. Open-system GRAPE (optimize with decoherence included)
2. Proper unit calibration to real hardware parameters

For now, we report only the closed-system results, which are scientifically valid.
