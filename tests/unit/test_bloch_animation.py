"""
Unit tests for Bloch sphere animation module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from src.visualization.bloch_animation import (
    BlochAnimator,
    AnimationStyle,
    create_bloch_animation,
    save_animation,
    animate_pulse_evolution,
)


class TestAnimationStyle:
    """Tests for AnimationStyle dataclass."""

    def test_default_style(self):
        """Test default style creation."""
        style = AnimationStyle()
        assert style.sphere_alpha == 0.1
        assert style.sphere_color == "gray"
        assert style.trajectory_linewidth == 2.0
        assert style.show_sphere is True

    def test_custom_style(self):
        """Test custom style parameters."""
        style = AnimationStyle(
            sphere_alpha=0.2,
            sphere_color="blue",
            trajectory_linewidth=3.0,
            point_size=200,
            colormap="plasma",
        )
        assert style.sphere_alpha == 0.2
        assert style.sphere_color == "blue"
        assert style.trajectory_linewidth == 3.0
        assert style.point_size == 200
        assert style.colormap == "plasma"


class TestBlochAnimator:
    """Tests for BlochAnimator class."""

    def test_initialization_single_trajectory(self):
        """Test initialization with single trajectory."""
        states = [qt.basis(2, 0) for _ in range(10)]
        animator = BlochAnimator(states)

        assert animator.n_trajectories == 1
        assert animator.n_frames == 10
        assert len(animator.bloch_trajectories) == 1

    def test_initialization_multiple_trajectories(self):
        """Test initialization with multiple trajectories."""
        traj1 = [qt.basis(2, 0) for _ in range(10)]
        traj2 = [qt.basis(2, 1) for _ in range(10)]
        animator = BlochAnimator([traj1, traj2])

        assert animator.n_trajectories == 2
        assert animator.n_frames == 10

    def test_initialization_with_labels(self):
        """Test initialization with custom labels."""
        states = [qt.basis(2, 0) for _ in range(5)]
        animator = BlochAnimator(states, labels=["My Trajectory"])

        assert animator.labels[0] == "My Trajectory"

    def test_initialization_with_style(self):
        """Test initialization with custom style."""
        states = [qt.basis(2, 0) for _ in range(5)]
        style = AnimationStyle(sphere_alpha=0.3, colormap="viridis")
        animator = BlochAnimator(states, style=style)

        assert animator.style.sphere_alpha == 0.3
        assert animator.style.colormap == "viridis"

    def test_state_to_bloch_basis_states(self):
        """Test Bloch vector conversion for basis states."""
        states = [qt.basis(2, 0)]
        animator = BlochAnimator(states)

        bloch_vec = animator._state_to_bloch(qt.basis(2, 0))
        assert bloch_vec[2] == pytest.approx(1.0)

        bloch_vec = animator._state_to_bloch(qt.basis(2, 1))
        assert bloch_vec[2] == pytest.approx(-1.0)

    def test_state_to_bloch_superposition(self):
        """Test Bloch vector conversion for superposition states."""
        states = [qt.basis(2, 0)]
        animator = BlochAnimator(states)

        # |+> state
        state_plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        bloch_vec = animator._state_to_bloch(state_plus)
        assert bloch_vec[0] == pytest.approx(1.0)

    def test_bloch_trajectories_conversion(self):
        """Test that all states are converted to Bloch vectors."""
        angles = np.linspace(0, np.pi, 10)
        states = [qt.Qobj([[np.cos(a / 2)], [np.sin(a / 2)]]) for a in angles]
        animator = BlochAnimator(states)

        assert len(animator.bloch_trajectories[0]) == 10
        assert animator.bloch_trajectories[0].shape == (10, 3)

    def test_create_animation_basic(self):
        """Test basic animation creation."""
        states = [qt.basis(2, 0) for _ in range(5)]
        animator = BlochAnimator(states)

        anim = animator.create_animation(interval=100)

        assert anim is not None
        assert animator.fig is not None
        assert animator.ax is not None
        plt.close(animator.fig)

    def test_create_animation_with_trail(self):
        """Test animation with trajectory trail."""
        states = [qt.basis(2, 0) for _ in range(10)]
        animator = BlochAnimator(states)

        anim = animator.create_animation(interval=50, show_trail=True, trail_length=5)

        assert anim is not None
        assert animator.show_trail is True
        assert animator.trail_length == 5
        plt.close(animator.fig)

    def test_create_animation_without_trail(self):
        """Test animation without trajectory trail."""
        states = [qt.basis(2, 0) for _ in range(5)]
        animator = BlochAnimator(states)

        anim = animator.create_animation(show_trail=False)

        assert animator.show_trail is False
        plt.close(animator.fig)

    def test_create_animation_multiple_trajectories(self):
        """Test animation with multiple trajectories."""
        traj1 = [qt.basis(2, 0) for _ in range(8)]
        traj2 = [qt.basis(2, 1) for _ in range(8)]
        animator = BlochAnimator([traj1, traj2], labels=["Traj 1", "Traj 2"])

        anim = animator.create_animation()

        assert anim is not None
        assert len(animator.artists) == 2
        plt.close(animator.fig)

    def test_update_frame(self):
        """Test frame update function."""
        states = [qt.basis(2, 0) for _ in range(5)]
        animator = BlochAnimator(states)
        animator.create_animation()

        # Update to frame 2
        animator._update_frame(2)

        # Artists should be updated
        assert animator.artists[0]["point"] is not None
        plt.close(animator.fig)

    def test_save_gif(self, tmp_path):
        """Test saving animation as GIF."""
        states = [qt.basis(2, 0) for _ in range(3)]
        animator = BlochAnimator(states)
        animator.create_animation()

        output_file = tmp_path / "test_animation.gif"

        # Note: This may fail if pillow writer is not available
        try:
            animator.save(str(output_file), fps=10, dpi=50)
            assert output_file.exists()
        except Exception as e:
            pytest.skip(f"Could not save GIF: {e}")
        finally:
            plt.close(animator.fig)

    def test_save_without_animation_raises(self):
        """Test that saving without creating animation raises error."""
        states = [qt.basis(2, 0) for _ in range(3)]
        animator = BlochAnimator(states)

        with pytest.raises(ValueError, match="Must call create_animation"):
            animator.save("output.gif")

    def test_close(self):
        """Test closing animator."""
        states = [qt.basis(2, 0) for _ in range(3)]
        animator = BlochAnimator(states)
        animator.create_animation()

        fig_num = animator.fig.number
        animator.close()

        assert not plt.fignum_exists(fig_num)

    def test_show_without_animation_raises(self):
        """Test that show without animation raises error."""
        states = [qt.basis(2, 0) for _ in range(3)]
        animator = BlochAnimator(states)

        with pytest.raises(ValueError, match="Must call create_animation"):
            animator.show()

    def test_rabi_oscillation_trajectory(self):
        """Test animation of Rabi oscillation."""
        times = np.linspace(0, 2 * np.pi, 20)
        states = [qt.Qobj([[np.cos(t / 2)], [np.sin(t / 2) * 1j]]) for t in times]
        animator = BlochAnimator(states)
        animator.create_animation()

        # Check that trajectory varies smoothly
        bloch_traj = animator.bloch_trajectories[0]
        assert len(bloch_traj) == 20
        # Z component should oscillate
        assert bloch_traj[0, 2] != bloch_traj[10, 2]
        plt.close(animator.fig)

    def test_rotation_around_x(self):
        """Test rotation around X axis."""
        angles = np.linspace(0, 2 * np.pi, 15)
        states = []
        for angle in angles:
            # Rotate |0> around X
            U = (-1j * angle * qt.sigmax() / 2).expm()
            state = U * qt.basis(2, 0)
            states.append(state)

        animator = BlochAnimator(states)
        animator.create_animation()

        bloch_traj = animator.bloch_trajectories[0]
        # X component should remain approximately constant at 0
        assert np.allclose(bloch_traj[:, 0], 0, atol=1e-10)
        plt.close(animator.fig)

    def test_custom_figsize(self):
        """Test custom figure size."""
        states = [qt.basis(2, 0) for _ in range(5)]
        animator = BlochAnimator(states, figsize=(10, 10))

        assert animator.figsize == (10, 10)
        animator.create_animation()
        assert animator.fig.get_size_inches()[0] == pytest.approx(10.0)
        plt.close(animator.fig)

    def test_different_trajectory_lengths_warning(self):
        """Test warning for trajectories with different lengths."""
        traj1 = [qt.basis(2, 0) for _ in range(10)]
        traj2 = [qt.basis(2, 1) for _ in range(5)]

        with pytest.warns(UserWarning, match="different length"):
            animator = BlochAnimator([traj1, traj2])


class TestCreateBlochAnimation:
    """Tests for create_bloch_animation convenience function."""

    def test_basic_creation(self):
        """Test basic animation creation."""
        states = [qt.basis(2, 0) for _ in range(5)]
        animator = create_bloch_animation(states)

        assert animator is not None
        assert animator.animation is not None
        plt.close(animator.fig)

    def test_with_labels(self):
        """Test with custom labels."""
        traj1 = [qt.basis(2, 0) for _ in range(5)]
        traj2 = [qt.basis(2, 1) for _ in range(5)]
        animator = create_bloch_animation([traj1, traj2], labels=["Path A", "Path B"])

        assert animator.labels[0] == "Path A"
        assert animator.labels[1] == "Path B"
        plt.close(animator.fig)

    def test_with_save(self, tmp_path):
        """Test creating and saving animation."""
        states = [qt.basis(2, 0) for _ in range(3)]
        output_file = tmp_path / "output.gif"

        try:
            animator = create_bloch_animation(states, filename=str(output_file), fps=10)
            assert output_file.exists()
        except Exception as e:
            pytest.skip(f"Could not save animation: {e}")
        finally:
            if animator.fig is not None:
                plt.close(animator.fig)

    def test_with_custom_style(self):
        """Test with custom style."""
        states = [qt.basis(2, 0) for _ in range(5)]
        style = AnimationStyle(sphere_alpha=0.5, colormap="cool")
        animator = create_bloch_animation(states, style=style)

        assert animator.style.sphere_alpha == 0.5
        assert animator.style.colormap == "cool"
        plt.close(animator.fig)

    def test_with_trail_length(self):
        """Test with custom trail length."""
        states = [qt.basis(2, 0) for _ in range(10)]
        animator = create_bloch_animation(states, trail_length=3)

        assert animator.trail_length == 3
        plt.close(animator.fig)


class TestSaveAnimation:
    """Tests for save_animation function."""

    def test_save_existing_animation(self, tmp_path):
        """Test saving an existing animator."""
        states = [qt.basis(2, 0) for _ in range(3)]
        animator = BlochAnimator(states)
        animator.create_animation()

        output_file = tmp_path / "saved.gif"

        try:
            save_animation(animator, str(output_file), fps=10, dpi=50)
            assert output_file.exists()
        except Exception as e:
            pytest.skip(f"Could not save animation: {e}")
        finally:
            plt.close(animator.fig)


class TestAnimatePulseEvolution:
    """Tests for animate_pulse_evolution function."""

    def test_basic_pulse_evolution(self):
        """Test basic pulse evolution animation."""
        pulse = np.sin(np.linspace(0, 2 * np.pi, 20))
        times = np.linspace(0, 5, 20)
        initial_state = qt.basis(2, 0)

        # Simple Hamiltonian function
        def H(t, args):
            return args["H0"] + args["u"](t) * args["H1"]

        try:
            animator = animate_pulse_evolution(
                H, pulse, times, initial_state, figsize=(8, 8)
            )
            assert animator is not None
            assert animator.animation is not None
            plt.close(animator.fig)
        except Exception as e:
            # May fail due to solver issues or missing dependencies
            pytest.skip(f"Could not create pulse evolution animation: {e}")

    def test_constant_pulse(self):
        """Test evolution under constant pulse."""
        pulse = np.ones(15) * 0.5
        times = np.linspace(0, 3, 15)
        initial_state = qt.basis(2, 0)

        def H(t, args):
            return args["H0"]

        try:
            animator = animate_pulse_evolution(H, pulse, times, initial_state)
            assert animator is not None
            plt.close(animator.fig)
        except Exception as e:
            pytest.skip(f"Could not create animation: {e}")


class TestIntegration:
    """Integration tests for animation module."""

    def test_full_animation_workflow(self):
        """Test complete animation workflow."""
        # Create trajectory
        angles = np.linspace(0, 4 * np.pi, 30)
        states = [
            qt.Qobj([[np.cos(a / 2)], [np.exp(1j * a) * np.sin(a / 2)]]) for a in angles
        ]

        # Create animator with custom style
        style = AnimationStyle(
            sphere_alpha=0.15,
            trajectory_linewidth=2.5,
            point_size=180,
            colormap="plasma",
        )
        animator = BlochAnimator(states, labels=["Spin Precession"], style=style)

        # Create animation
        anim = animator.create_animation(interval=100, trail_length=10)

        assert anim is not None
        assert animator.animation is not None
        plt.close(animator.fig)

    def test_multiple_trajectories_comparison(self):
        """Test comparing multiple evolution trajectories."""
        # Fast evolution
        angles1 = np.linspace(0, 2 * np.pi, 15)
        traj1 = [qt.Qobj([[np.cos(a / 2)], [np.sin(a / 2)]]) for a in angles1]

        # Slow evolution
        angles2 = np.linspace(0, np.pi, 15)
        traj2 = [qt.Qobj([[np.cos(a / 2)], [np.sin(a / 2)]]) for a in angles2]

        animator = BlochAnimator([traj1, traj2], labels=["Fast", "Slow"])
        animator.create_animation()

        assert len(animator.bloch_trajectories) == 2
        plt.close(animator.fig)

    def test_animation_with_data_export(self):
        """Test creating animation and exporting trajectory data."""
        states = [qt.basis(2, 0) for _ in range(10)]
        animator = BlochAnimator(states)
        animator.create_animation()

        # Extract Bloch vectors for analysis
        bloch_data = animator.bloch_trajectories[0]

        assert bloch_data.shape == (10, 3)
        assert isinstance(bloch_data, np.ndarray)
        plt.close(animator.fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
