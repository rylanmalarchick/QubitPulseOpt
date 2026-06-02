"""
Reinforcement Learning Environment for Quantum Feedback Control
===============================================================

This module implements an OpenAI Gymnasium environment for training RL agents
to control quantum systems. It wraps the Stochastic Master Equation (SME)
solver to provide a standard RL interface:

    Observation: Measurement record history [I(t), I(t-dt), ...]
    Action: Control pulse amplitude Ω(t)
    Reward: Fidelity to target state (minus penalty for control energy)

Physical Model:
---------------
The environment simulates a single qubit under continuous weak measurement.
The agent acts as the FPGA controller, receiving noisy voltage signals and
deciding on real-time feedback pulses.

Dependencies:
-------------
- gymnasium
- numpy
- qutip
- stable_baselines3 (for training, optional import)

Author: Orchestrator Agent
Date: 2026-01-20
"""

import gymnasium as gym
import numpy as np
import qutip as qt
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from .hamiltonian.stochastic import MeasurementParams
from .hamiltonian.lindblad import DecoherenceParams

class QuantumStabilizationEnv(gym.Env):
    """
    RL Environment for Qubit Stabilization.
    
    Goal: Keep the qubit in the |0> state (or target state) despite T1/T2 noise
    and continuous measurement backaction.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        dt: float = 1e-3,
        max_steps: int = 100,
        decoherence: Optional[DecoherenceParams] = None,
        measurement: Optional[MeasurementParams] = None
    ):
        super().__init__()
        
        # Physics Parameters
        self.dt = dt
        self.max_steps = max_steps
        self.current_step = 0
        
        # Default Physics
        if decoherence is None:
            decoherence = DecoherenceParams(T1=10.0, T2=5.0)
        self.decoherence = decoherence
        
        if measurement is None:
            measurement = MeasurementParams(efficiency=1.0, strength=5.0)
        self.measurement = measurement
        
        # System Operators
        self.sigma_z = qt.sigmaz()
        self.sigma_x = qt.sigmax()
        self.target_state = qt.basis(2, 0) # Target |0>
        self.current_rho = self.target_state * self.target_state.dag()
        
        # Action Space: Control Amplitude Ω_x (continuous)
        # We clip it to [-10, 10] MHz
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(1,), dtype=np.float32
        )
        
        # Observation Space: 
        # [Measurement_Current, Measurement_Prev, Cos(Phase), Sin(Phase)]
        # For now, let's keep it simple: [Measurement Output]
        # Real-world FPGA would have a window.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        
        # Solver Initialization
        # We need to construct the SME update matrices efficiently
        # Pre-computing operators for dt step
        self._init_operators()

    def _init_operators(self):
        """Pre-compute operators for fast stepping."""
        # Stochastic propagator is hard to pre-compute exactly due to dW
        # We will use a simplified Euler-Maruyama step in the 'step' function
        pass

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset Physics
        self.current_step = 0
        self.current_rho = self.target_state * self.target_state.dag()
        
        # Initial Observation (No measurement yet, return 0)
        observation = np.array([0.0], dtype=np.float32)
        
        return observation, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step of the simulation.
        
        1. Apply Control Hamiltonian (Unitary)
        2. Apply Dissipation (Lindblad)
        3. Apply Measurement Backaction (Stochastic)
        4. Compute Reward
        """
        self.current_step += 1
        control_amp = float(action[0])
        
        # 1. Generate Noise (Wiener Process)
        # Use the Gymnasium-managed RNG so reset(seed=...) makes episodes
        # reproducible (required by the Gymnasium contract).
        dW = self.np_random.normal(0, np.sqrt(self.dt))
        
        # 2. Physics Update (Euler-Maruyama Step for SME)
        # dρ = -i[H, ρ]dt + D[L]ρ dt + H[M]ρ dW
        
        # Hamiltonian H = H0 + u(t)Hx
        H_drift = 0.5 * 0.0 * self.sigma_z # Zero detuning for now
        H_ctrl = control_amp * self.sigma_x
        H_total = H_drift + H_ctrl
        
        # Commutator term: -i[H, ρ]
        comm = -1j * (H_total * self.current_rho - self.current_rho * H_total)
        
        # Dissipation (T1)
        gamma1 = 1.0 / self.decoherence.T1
        L1 = np.sqrt(gamma1) * qt.destroy(2)
        D_L1 = L1 * self.current_rho * L1.dag() - 0.5 * (L1.dag() * L1 * self.current_rho + self.current_rho * L1.dag() * L1)
        
        # Measurement Backaction
        # M = √κ σ_z
        kappa = self.measurement.strength
        M = np.sqrt(kappa) * self.sigma_z
        
        # Innovation term H[M]ρ
        # H[c]ρ = cρ + ρc† - Tr(cρ + ρc†)ρ
        # For Hermitian M: Mρ + ρM - 2<M>ρ
        exp_M = (M * self.current_rho).tr() # Expectation value <M>
        H_M = (M * self.current_rho + self.current_rho * M) - 2 * exp_M * self.current_rho
        
        # Update Density Matrix
        drho = (comm + D_L1) * self.dt + H_M * dW
        self.current_rho = self.current_rho + drho
        
        # Re-normalize (numerical stability)
        self.current_rho = self.current_rho / self.current_rho.tr()
        
        # 3. Compute Measurement Output (Observation)
        # dI = <M>dt + dW
        meas_signal = np.real(exp_M) + dW/self.dt
        
        # 4. Compute Reward
        # Fidelity to Target |0>
        fidelity = qt.fidelity(self.current_rho, self.target_state)**2
        
        # Reward = Fidelity - ControlCost
        reward = fidelity - 0.01 * (control_amp**2)
        
        # 5. Check Termination
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return (
            np.array([meas_signal], dtype=np.float32), 
            reward, 
            terminated, 
            truncated, 
            {"fidelity": fidelity}
        )

    def render(self):
        """Optional visualization."""
        pass
