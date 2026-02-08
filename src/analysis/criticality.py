"""
Quantum Criticality Detector: Financial Warning Indicators for Quantum Errors
=============================================================================

"As in markets, so in qubits."

This module adapts statistical mechanics indicators used in financial crisis
prediction (Sornette, Bouchaud) to detect "Phase Slips" and "Decoherence Events"
in quantum trajectories.

Theory:
-------
A quantum error ($T_1$ decay or Phase Slip) is a "critical transition" in the
system state. Near this transition, we expect:
1.  **Critical Slowing Down**: Autocorrelation lag increases.
2.  **Susceptibility Divergence**: Variance (VIX) spikes.
3.  **Fat Tails**: Kurtosis increases as non-Gaussian noise dominates.

This module provides the `CriticalityMonitor` to analyze `I(t)` traces in real-time.

Author: Orchestrator Agent
Date: 2026-01-20
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class CriticalityMetrics:
    """Snapshot of criticality indicators at time t."""
    variance: float       # "Quantum VIX"
    autocorr: float       # Critical Slowing Down
    kurtosis: float       # Fat Tails
    score: float          # Composite 0-100 score

class CriticalityMonitor:
    """
    Real-time analyzer for quantum measurement trajectories.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.history = []
        
    def update(self, measurement_val: float) -> Optional[CriticalityMetrics]:
        """
        Push new measurement I(t) and calculate metrics.
        Returns None if history < window_size.
        """
        self.history.append(measurement_val)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        if len(self.history) < self.window_size:
            return None
            
        data = np.array(self.history)
        
        # 1. Variance ("Quantum VIX")
        # Diverges near T1 event (telegraph noise increases)
        variance = np.var(data)
        
        # 2. Autocorrelation (Critical Slowing Down)
        # Lag-1 autocorrelation. 
        # Warning: If |1> state is stable, ac -> 1. 
        # If noise increases, ac drops? Or increases?
        # In stat mech, AC -> 1 near critical point (slowing down).
        ac = self._autocorr(data, lag=1)
        
        # 3. Kurtosis (Fat Tails)
        # Normal Gaussian noise = 0 (Fisher).
        # Jumps/Slips = High Kurtosis.
        kurt = stats.kurtosis(data)
        
        # Composite Score (Heuristic)
        # Normalize roughly based on typical homodyne noise
        # Var ~ 1/dt. 
        score = (
            (variance * 10) + 
            (abs(ac) * 20) + 
            (max(0, kurt) * 5)
        )
        score = min(100.0, max(0.0, score))
        
        return CriticalityMetrics(
            variance=variance,
            autocorr=ac,
            kurtosis=kurt,
            score=score
        )
        
    def _autocorr(self, x: np.ndarray, lag: int = 1) -> float:
        """Compute lag-k autocorrelation."""
        if len(x) <= lag: return 0.0
        y1 = x[:-lag]
        y2 = x[lag:]
        if np.std(y1) == 0 or np.std(y2) == 0:
            return 0.0
        return np.corrcoef(y1, y2)[0, 1]

def analyze_trajectory(times: np.ndarray, signal: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Post-processing analysis of a full trajectory.
    """
    monitor = CriticalityMonitor(window_size=50)
    results = {
        "variance": [],
        "autocorr": [],
        "kurtosis": [],
        "score": []
    }
    
    # Pad beginning
    pad = [np.nan] * 50
    
    for val in signal:
        metrics = monitor.update(val)
        if metrics:
            results["variance"].append(metrics.variance)
            results["autocorr"].append(metrics.autocorr)
            results["kurtosis"].append(metrics.kurtosis)
            results["score"].append(metrics.score)
    
    # Convert to arrays and pad
    for k in results:
        results[k] = np.array([np.nan]*49 + results[k])
        
    return results
