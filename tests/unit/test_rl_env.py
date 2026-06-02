"""
Unit tests for the reinforcement-learning environment
(src/rl_env.py: QuantumStabilizationEnv).

These exercise the Gymnasium contract (spaces, reset/step signatures, episode
termination) and the reward semantics (fidelity reward minus control-energy
penalty). The environment integrates a simplified Euler-Maruyama SME step; the
deep physics is covered in test_stochastic.py, so here we focus on the RL API.
"""

import warnings

import numpy as np
import pytest
import qutip as qt

gym = pytest.importorskip("gymnasium")

from src.rl_env import QuantumStabilizationEnv
from src.hamiltonian.lindblad import DecoherenceParams
from src.hamiltonian.stochastic import MeasurementParams


class TestConstruction:
    def test_action_space(self):
        env = QuantumStabilizationEnv()
        assert env.action_space.shape == (1,)
        assert np.isclose(env.action_space.low[0], -10.0)
        assert np.isclose(env.action_space.high[0], 10.0)

    def test_observation_space(self):
        env = QuantumStabilizationEnv()
        assert env.observation_space.shape == (1,)

    def test_custom_physics(self):
        dec = DecoherenceParams(T1=15.0, T2=10.0)
        meas = MeasurementParams(strength=3.0)
        env = QuantumStabilizationEnv(dt=0.02, max_steps=50, decoherence=dec, measurement=meas)
        assert env.dt == 0.02
        assert env.max_steps == 50
        assert env.decoherence.T1 == 15.0


class TestReset:
    def test_reset_returns_obs_and_info(self):
        env = QuantumStabilizationEnv()
        obs, info = env.reset(seed=0)
        assert obs.shape == (1,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_reset_state_is_ground(self):
        env = QuantumStabilizationEnv()
        env.reset(seed=0)
        assert env.current_step == 0
        # current_rho should be |0><0|.
        assert abs(env.current_rho.tr() - 1.0) < 1e-12
        assert qt.fidelity(env.current_rho, qt.basis(2, 0)) ** 2 == pytest.approx(1.0, abs=1e-9)


class TestStep:
    def test_step_signature_and_types(self):
        env = QuantumStabilizationEnv()
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        assert obs.shape == (1,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "fidelity" in info

    def test_step_increments_counter(self):
        env = QuantumStabilizationEnv()
        env.reset(seed=0)
        env.step(np.array([0.0], dtype=np.float32))
        assert env.current_step == 1

    def test_state_remains_normalized(self):
        env = QuantumStabilizationEnv()
        env.reset(seed=1)
        for _ in range(20):
            env.step(np.array([1.0], dtype=np.float32))
            assert abs(env.current_rho.tr() - 1.0) < 1e-9

    def test_fidelity_bounded(self):
        env = QuantumStabilizationEnv()
        env.reset(seed=2)
        for _ in range(30):
            _, _, _, _, info = env.step(np.array([2.0], dtype=np.float32))
            assert -1e-9 <= info["fidelity"] <= 1.0 + 1e-9

    def test_truncation_at_max_steps(self):
        env = QuantumStabilizationEnv(max_steps=5)
        env.reset(seed=0)
        truncated = False
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
        assert truncated is True


class TestRewardSemantics:
    def test_control_energy_is_penalized(self):
        # From the same state and noise (same reset seed), a large control costs
        # more reward than no control (reward = fidelity - 0.01 * amp^2).
        env_zero = QuantumStabilizationEnv()
        env_zero.reset(seed=0)
        _, reward_zero, _, _, _ = env_zero.step(np.array([0.0], dtype=np.float32))

        env_big = QuantumStabilizationEnv()
        env_big.reset(seed=0)
        _, reward_big, _, _, _ = env_big.step(np.array([9.0], dtype=np.float32))

        assert reward_zero > reward_big

    def test_deterministic_given_reset_seed(self):
        env1 = QuantumStabilizationEnv()
        env1.reset(seed=0)
        obs1, r1, _, _, _ = env1.step(np.array([3.0], dtype=np.float32))

        env2 = QuantumStabilizationEnv()
        env2.reset(seed=0)
        obs2, r2, _, _, _ = env2.step(np.array([3.0], dtype=np.float32))

        assert np.allclose(obs1, obs2)
        assert r1 == pytest.approx(r2)


class TestGymnasiumContract:
    def test_passes_env_checker(self):
        # Gymnasium's checker validates the reset/step API and spaces. It emits
        # a UserWarning about the infinite observation bounds (intentional here),
        # which we silence so it does not trip filterwarnings=error; genuine API
        # violations still raise.
        from gymnasium.utils.env_checker import check_env

        env = QuantumStabilizationEnv(max_steps=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_env(env, skip_render_check=True)
