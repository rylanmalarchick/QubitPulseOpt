"""
Unit tests for the PPO training entry point (src/train_agent.py).

``build_env`` is dependency-light and always tested. The training/eval path
requires stable-baselines3 (heavy: pulls in PyTorch); those tests run a tiny
smoke train when sb3 is installed and are honestly skipped otherwise.
"""

import importlib.util
import warnings

import numpy as np
import pytest

from src.train_agent import build_env, train, evaluate
from src.rl_env import QuantumStabilizationEnv

HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None
sb3_required = pytest.mark.skipif(
    not HAS_SB3, reason="stable_baselines3 not installed"
)


class TestBuildEnv:
    def test_returns_environment(self):
        env = build_env(dt=0.02, max_steps=30, T1=15.0, T2=10.0, strength=3.0)
        assert isinstance(env, QuantumStabilizationEnv)
        assert env.dt == 0.02
        assert env.max_steps == 30
        assert env.decoherence.T1 == 15.0
        assert env.measurement.strength == 3.0

    def test_default_env_is_runnable(self):
        env = build_env(max_steps=5)
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)


@sb3_required
class TestTraining:
    # sb3's check_env emits an advisory UserWarning that the action space is not
    # the recommended symmetric [-1, 1] Box (ours is [-10, 10] MHz). That is a
    # tuning tip, not an error; silence it so filterwarnings=error does not trip.
    def test_smoke_train_produces_usable_policy(self):
        # Tiny training run: just enough to exercise the PPO loop end-to-end.
        env = build_env(max_steps=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, env = train(
                total_timesteps=128,
                env=env,
                n_steps=64,
                batch_size=32,
                seed=0,
                verbose=0,
            )
        obs, _ = env.reset(seed=0)
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1,)
        assert env.action_space.contains(action)

    def test_evaluate_returns_finite_stats(self):
        env = build_env(max_steps=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, env = train(
                total_timesteps=128, env=env, n_steps=64, batch_size=32, seed=0
            )
            mean_reward, std_reward = evaluate(model, env, n_eval_episodes=2)
        assert np.isfinite(mean_reward)
        assert std_reward >= 0.0
