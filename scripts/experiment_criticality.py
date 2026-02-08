"""
Experiment: Link the Watchdog
=============================

Objective: Verify that the CriticalityMonitor (Quantum VIX) spikes *before*
the qubit fidelity completely collapses.

Method:
1. Initialize QuantumStabilizationEnv with standard T1/T2 noise.
2. Run a trajectory with NO control (allow decay).
3. Record Measurement I(t) and Fidelity F(t).
4. Run CriticalityMonitor on I(t).
5. Correlate 'Criticality Score' with 'Fidelity Drop'.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rl_env import QuantumStabilizationEnv
from src.analysis.criticality import analyze_trajectory
from src.hamiltonian.lindblad import DecoherenceParams
from src.hamiltonian.stochastic import MeasurementParams

def run_experiment():
    print("Initializing Experiment: Criticality Watchdog...")
    
    # 1. Setup Environment
    # We want a reasonably fast decay to observe the transition
    T1 = 5.0  # units of 1/gamma
    T2 = 5.0
    dt = 0.05
    max_steps = 200 # Total time = 10.0 (2x T1)
    
    env = QuantumStabilizationEnv(
        dt=dt,
        max_steps=max_steps,
        decoherence=DecoherenceParams(T1=T1, T2=T2),
        measurement=MeasurementParams(strength=2.0) # Strong measurement
    )
    
    obs, _ = env.reset()
    
    # OVERRIDE: Start in Excited State |1> to observe T1 decay
    import qutip as qt
    env.target_state = qt.basis(2, 1) # Target is |1> (Excited)
    env.current_rho = env.target_state * env.target_state.dag()
    print("DEBUG: Overridden initial state to |1> (Excited) to simulate T1 decay.")
    
    # 2. Run Trajectory (No Control)
    measurements = []
    fidelities = []
    
    print(f"Running simulation for {max_steps} steps (dt={dt})...")
    
    for _ in range(max_steps):
        # Action = 0 (No feedback)
        action = np.array([0.0], dtype=np.float32)
        
        # Step
        next_obs, reward, term, trunc, info = env.step(action)
        
        measurements.append(next_obs[0])
        fidelities.append(info['fidelity'])
        
        if term or trunc:
            break
            
    measurements = np.array(measurements)
    fidelities = np.array(fidelities)
    
    # 3. Analyze Criticality
    print("Analyzing trajectory...")
    crit_stats = analyze_trajectory(np.arange(len(measurements))*dt, measurements)
    scores = crit_stats['score']
    variances = crit_stats['variance']
    
    # 4. Report Results
    print("\nResults Table (Sampled):")
    print(f"{'Step':<5} | {'Time':<5} | {'Fidelity':<8} | {'CritScore':<10} | {'Variance':<8}")
    print("-" * 50)
    
    spike_detected = False
    warning_time = -1
    failure_time = -1
    
    for i in range(0, len(measurements), 10): # Print every 10th step
        t = i * dt
        f = fidelities[i]
        s = scores[i]
        v = variances[i]
        
        marker = ""
        if s > 50 and not spike_detected:
            marker = "<-- WARNING SPIKE"
            spike_detected = True
            warning_time = t
            
        if f < 0.9 and failure_time < 0:
            failure_time = t
            if spike_detected:
                 marker += " (Predicted!)"
            else:
                 marker += " (Unpredicted)"
                 
        print(f"{i:<5} | {t:<5.2f} | {f:<8.4f} | {s:<10.2f} | {v:<8.4f} {marker}")

    # Summary
    print("\nAnalysis:")
    if spike_detected:
        print(f"SUCCESS: Watchdog spiked (Score > 50) at t={warning_time:.2f}")
    else:
        print("FAILURE: No warning spike detected.")
        
    if failure_time > 0:
        print(f"System Failed (F < 0.9) at t={failure_time:.2f}")
        if spike_detected and warning_time < failure_time:
            lead_time = failure_time - warning_time
            print(f"Early Warning Lead Time: {lead_time:.2f}s")
        elif spike_detected:
            print("Warning came too late.")
    
    # Optional: Save plot data
    np.savez("experiment_criticality_data.npz", 
             measurements=measurements, 
             fidelities=fidelities, 
             scores=scores)
    print("\nData saved to experiment_criticality_data.npz")

if __name__ == "__main__":
    run_experiment()
