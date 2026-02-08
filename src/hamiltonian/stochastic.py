"""
Stochastic Master Equation (SME) for Quantum Feedback Control
=============================================================

This module implements the Stochastic Master Equation (SME) for simulating
individual quantum trajectories under continuous weak measurement. This is the
foundation for simulating real-time quantum feedback control.

Physical Context:
-----------------
While the Lindblad master equation describes the *average* evolution of an
ensemble, the SME describes the evolution of a *single* system conditioned on
a measurement record I(t).

    dρ = -i[H, ρ]dt + L[ρ]dt + √η H[ρ]dW

where:
- dW is the Wiener process (Gaussian white noise, dW ~ √dt)
- H[ρ] is the measurement superoperator (backaction)
- η is the measurement efficiency

This requires Itô calculus to simulate correctly.

Classes:
--------
- StochasticEvolution: Simulates single trajectories using homodyne/heterodyne detection.

Dependencies:
-------------
- QuTiP (stochastic solvers)
- NumPy
- SciPy

Author: Orchestrator Agent (Ghost of Itô)
Date: 2026-01-20
"""

import numpy as np
import qutip as qt
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from .lindblad import DecoherenceParams

@dataclass
class MeasurementParams:
    """
    Parameters for continuous weak measurement.
    
    Attributes
    ----------
    efficiency : float
        Measurement efficiency η ∈ [0, 1].
        η=1: Perfect measurement (pure state remains pure).
        η=0: No information (Lindblad evolution).
    strength : float
        Measurement strength κ (rate of information gain).
    axis : str
        Measurement axis ('x', 'y', 'z').
    method : str
        'homodyne' (measures one quadrature) or 'heterodyne' (measures both).
    """
    efficiency: float = 1.0
    strength: float = 1.0
    axis: str = 'z'
    method: str = 'homodyne'

class StochasticEvolution:
    """
    Stochastic Master Equation solver for quantum trajectories.
    
    Simulates:
        dρ_c = -i[H, ρ_c]dt + D[L]ρ_c dt + √η H[L]ρ_c dW
        
    where ρ_c is the conditioned density matrix.
    """
    
    def __init__(
        self,
        H: Union[qt.Qobj, list],
        decoherence: DecoherenceParams,
        measurement: MeasurementParams
    ):
        self.H = H
        self.decoherence = decoherence
        self.measurement = measurement
        
        # System dimension
        self.dim = H[0].shape[0] if isinstance(H, list) else H.shape[0]
        
        # Build operators
        self.c_ops = self._build_collapse_ops()
        self.sc_ops = self._build_stochastic_ops()
        
    def _build_collapse_ops(self) -> List[qt.Qobj]:
        """Standard environmental noise (unmonitored)."""
        # Re-use logic from Lindblad, but exclude the monitored channel
        # For simplicity, we assume T1/T2 are separate from the measurement channel
        gamma1 = 1.0 / self.decoherence.T1
        c_ops = [np.sqrt(gamma1) * qt.destroy(self.dim)]
        
        if self.decoherence.T2:
            gamma2 = 1.0 / self.decoherence.T2
            gamma_phi = gamma2 - gamma1/2
            if gamma_phi > 0:
                c_ops.append(np.sqrt(gamma_phi) * qt.sigmaz())
                
        return c_ops

    def _build_stochastic_ops(self) -> List[qt.Qobj]:
        """Measurement operators (monitored channels)."""
        # Measurement operator M associated with strength κ
        # L_meas = √κ * σ_axis
        kappa = self.measurement.strength
        
        if self.measurement.axis == 'z':
            op = qt.sigmaz()
        elif self.measurement.axis == 'x':
            op = qt.sigmax()
        elif self.measurement.axis == 'y':
            op = qt.sigmay()
        else:
            raise ValueError(f"Invalid axis: {self.measurement.axis}")
            
        return [np.sqrt(kappa) * op]

    def evolve(
        self,
        rho0: qt.Qobj,
        times: np.ndarray,
        n_trajectories: int = 1,
        seed: Optional[int] = None
    ) -> qt.solver.Result:
        """
        Simulate stochastic trajectories.
        
        Parameters
        ----------
        rho0 : Initial state
        times : Time array
        n_trajectories : Number of trajectories to average/store
        seed : Random seed
        
        Returns
        -------
        result : QuTiP Result object
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Select solver based on method
        if self.measurement.method == 'homodyne':
            # smesolve for homodyne
            return qt.smesolve(
                self.H, rho0, times,
                c_ops=self.c_ops,
                sc_ops=self.sc_ops,
                e_ops=[qt.sigmaz(), qt.sigmax(), qt.sigmay()],
                ntraj=n_trajectories,
                nsubsteps=10,  # Important for Itô convergence
                store_measurement=True,
                method='homodyne'
            )
        elif self.measurement.method == 'photodetection':
             # mcsolve for jump trajectories (if applicable)
             # But usually feedback uses diffusive (smesolve)
             pass
        
        raise NotImplementedError(f"Method {self.measurement.method} not implemented")

    def simulate_feedback_loop(
        self,
        rho0: qt.Qobj,
        times: np.ndarray,
        feedback_func: callable
    ):
        """
        Simulate explicit feedback loop where H depends on measurement record.
        
        This requires manual stepping because QuTiP's builtin feedback
        is limited.
        
        TODO: Implement manual Euler-Maruyama stepper with feedback.
        """
        pass
