"""
Bloch sphere animation tools for visualizing quantum state evolution.

This module provides tools for creating animations of quantum state trajectories
on the Bloch sphere, with support for multiple trajectories, custom styling,
and export to GIF/MP4 formats.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import qutip as qt
from typing import Optional, List, Tuple, Union, Callable
from dataclasses import dataclass
import warnings


@dataclass
class AnimationStyle:
    """Styling options for Bloch sphere animations."""

    sphere_alpha: float = 0.1
    sphere_color: str = "gray"
    trajectory_linewidth: float = 2.0
    trajectory_alpha: float = 0.8
    point_size: int = 150
    point_alpha: float = 1.0
    show_axes: bool = True
    show_sphere: bool = True
    colormap: str = "viridis"
    background_color: str = "white"


class BlochAnimator:
    """
    Create animations of quantum state evolution on the Bloch sphere.

    This class handles the creation of animations showing how quantum states
    evolve over time, with support for multiple trajectories, custom styling,
    and various export formats.

    Examples
    --------
    >>> # Animate a Rabi oscillation
    >>> times = np.linspace(0, 2*np.pi, 100)
    >>> states = [qt.Qobj([[np.cos(t/2)], [np.sin(t/2)*1j]]) for t in times]
    >>> animator = BlochAnimator(states)
    >>> anim = animator.create_animation(interval=50)
    >>> animator.save('rabi.gif', fps=20)

    >>> # Multiple trajectories
    >>> animator = BlochAnimator([states1, states2], labels=['Path 1', 'Path 2'])
    >>> animator.create_animation()
    """

    def __init__(
        self,
        trajectories: Union[List[qt.Qobj], List[List[qt.Qobj]]],
        labels: Optional[List[str]] = None,
        style: Optional[AnimationStyle] = None,
        figsize: Tuple[int, int] = (8, 8),
    ):
        """
        Initialize Bloch sphere animator.

        Parameters
        ----------
        trajectories : list of Qobj or list of list of Qobj
            Single trajectory (list of states) or multiple trajectories
        labels : list of str, optional
            Labels for each trajectory
        style : AnimationStyle, optional
            Styling options (defaults to AnimationStyle())
        figsize : tuple
            Figure size (width, height)
        """
        # Normalize input to list of trajectories
        if isinstance(trajectories[0], qt.Qobj):
            # Single trajectory
            self.trajectories = [trajectories]
        else:
            # Multiple trajectories
            self.trajectories = trajectories

        self.n_trajectories = len(self.trajectories)
        self.labels = labels or [
            f"Trajectory {i + 1}" for i in range(self.n_trajectories)
        ]
        self.style = style or AnimationStyle()
        self.figsize = figsize

        # Animation state
        self.fig = None
        self.ax = None
        self.animation = None
        self.artists = []

        # Convert states to Bloch vectors
        self.bloch_trajectories = []
        for traj in self.trajectories:
            bloch_vecs = np.array([self._state_to_bloch(state) for state in traj])
            self.bloch_trajectories.append(bloch_vecs)

        # Check all trajectories have same length
        self.n_frames = len(self.bloch_trajectories[0])
        for i, traj in enumerate(self.bloch_trajectories):
            if len(traj) != self.n_frames:
                warnings.warn(
                    f"Trajectory {i} has different length ({len(traj)} vs {self.n_frames}). "
                    "This may cause animation issues."
                )

    def create_animation(
        self,
        interval: int = 50,
        trail_length: Optional[int] = None,
        show_trail: bool = True,
    ) -> FuncAnimation:
        """
        Create the animation.

        Parameters
        ----------
        interval : int
            Time between frames in milliseconds
        trail_length : int, optional
            Number of previous points to show (None = show all)
        show_trail : bool
            Whether to show the trajectory trail

        Returns
        -------
        animation : FuncAnimation
        """
        self.trail_length = trail_length
        self.show_trail = show_trail

        # Setup figure
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.patch.set_facecolor(self.style.background_color)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor(self.style.background_color)

        # Draw static elements
        if self.style.show_sphere:
            self._draw_bloch_sphere()

        # Setup artists for each trajectory
        cmap = plt.colormaps.get_cmap(self.style.colormap)
        colors = cmap(np.linspace(0, 1, self.n_trajectories))

        self.artists = []
        for i in range(self.n_trajectories):
            artist_dict = {
                "color": colors[i],
                "label": self.labels[i],
                "trail": None,
                "point": None,
                "vector": None,
            }
            self.artists.append(artist_dict)

        # Add legend
        if self.n_trajectories > 1:
            # Create dummy artists for legend
            for i, artist in enumerate(self.artists):
                self.ax.plot(
                    [],
                    [],
                    "o",
                    color=artist["color"],
                    label=artist["label"],
                    markersize=8,
                )
            self.ax.legend(loc="upper left")

        # Configure axes
        self._configure_axes()

        # Create animation
        self.animation = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self.n_frames,
            interval=interval,
            blit=False,
            repeat=True,
        )

        return self.animation

    def _update_frame(self, frame: int):
        """Update animation frame."""
        # Remove previous artists
        for artist_dict in self.artists:
            if artist_dict["trail"] is not None:
                artist_dict["trail"].remove()
            if artist_dict["point"] is not None:
                artist_dict["point"].remove()
            if artist_dict["vector"] is not None:
                artist_dict["vector"].remove()

        # Update each trajectory
        for i, (bloch_traj, artist_dict) in enumerate(
            zip(self.bloch_trajectories, self.artists)
        ):
            color = artist_dict["color"]

            # Determine trail start
            if self.trail_length is not None:
                start_idx = max(0, frame - self.trail_length)
            else:
                start_idx = 0

            # Draw trail
            if self.show_trail and frame > 0:
                trail_data = bloch_traj[start_idx : frame + 1]
                artist_dict["trail"] = self.ax.plot(
                    trail_data[:, 0],
                    trail_data[:, 1],
                    trail_data[:, 2],
                    color=color,
                    linewidth=self.style.trajectory_linewidth,
                    alpha=self.style.trajectory_alpha,
                )[0]

            # Draw current point
            current_pos = bloch_traj[frame]
            artist_dict["point"] = self.ax.scatter(
                current_pos[0],
                current_pos[1],
                current_pos[2],
                color=color,
                s=self.style.point_size,
                alpha=self.style.point_alpha,
                edgecolors="black",
                linewidths=1.5,
            )

            # Draw vector from origin
            artist_dict["vector"] = self.ax.quiver(
                0,
                0,
                0,
                current_pos[0],
                current_pos[1],
                current_pos[2],
                color=color,
                arrow_length_ratio=0.15,
                linewidth=1.5,
                alpha=0.6,
            )

        # Update title with frame number
        self.ax.set_title(f"Frame {frame + 1}/{self.n_frames}", fontsize=12, pad=10)

        return []

    def save(
        self,
        filename: str,
        fps: int = 20,
        dpi: int = 100,
        writer: Optional[str] = None,
    ):
        """
        Save animation to file.

        Parameters
        ----------
        filename : str
            Output filename (extension determines format: .gif, .mp4, etc.)
        fps : int
            Frames per second
        dpi : int
            Resolution in dots per inch
        writer : str, optional
            Writer to use ('pillow' for GIF, 'ffmpeg' for MP4)
            If None, automatically determined from filename
        """
        if self.animation is None:
            raise ValueError("Must call create_animation() before save()")

        # Determine writer from filename if not specified
        if writer is None:
            if filename.endswith(".gif"):
                writer = "pillow"
            elif filename.endswith(".mp4"):
                writer = "ffmpeg"
            else:
                # Default to pillow
                writer = "pillow"
                warnings.warn(
                    f"Could not determine format from filename '{filename}'. "
                    "Using GIF format."
                )

        # Setup writer
        if writer == "pillow":
            writer_obj = PillowWriter(fps=fps)
        elif writer == "ffmpeg":
            writer_obj = FFMpegWriter(fps=fps)
        else:
            raise ValueError(f"Unknown writer: {writer}")

        # Save
        print(f"Saving animation to {filename}...")
        self.animation.save(filename, writer=writer_obj, dpi=dpi)
        print(f"Animation saved successfully!")

    def _draw_bloch_sphere(self):
        """Draw the Bloch sphere surface and axes."""
        # Sphere surface
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        self.ax.plot_surface(
            x,
            y,
            z,
            color=self.style.sphere_color,
            alpha=self.style.sphere_alpha,
            linewidth=0,
        )

        # Equator and meridians
        theta = np.linspace(0, 2 * np.pi, 100)
        self.ax.plot(np.cos(theta), np.sin(theta), 0, "k-", alpha=0.2, linewidth=0.8)
        self.ax.plot(np.cos(theta), 0, np.sin(theta), "k-", alpha=0.2, linewidth=0.8)
        self.ax.plot(0, np.cos(theta), np.sin(theta), "k-", alpha=0.2, linewidth=0.8)

        if self.style.show_axes:
            # Coordinate axes
            self.ax.plot([0, 1.3], [0, 0], [0, 0], "k-", linewidth=1.5, alpha=0.5)
            self.ax.plot([0, 0], [0, 1.3], [0, 0], "k-", linewidth=1.5, alpha=0.5)
            self.ax.plot([0, 0], [0, 0], [0, 1.3], "k-", linewidth=1.5, alpha=0.5)

            # Axis labels
            self.ax.text(1.4, 0, 0, "X", fontsize=14, fontweight="bold")
            self.ax.text(0, 1.4, 0, "Y", fontsize=14, fontweight="bold")
            self.ax.text(0, 0, 1.4, "Z", fontsize=14, fontweight="bold")

    def _configure_axes(self):
        """Configure 3D axes appearance."""
        self.ax.set_xlabel("X", fontsize=11)
        self.ax.set_ylabel("Y", fontsize=11)
        self.ax.set_zlabel("Z", fontsize=11)

        # Set equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlim([-1.2, 1.2])
        self.ax.set_ylim([-1.2, 1.2])
        self.ax.set_zlim([-1.2, 1.2])

        # Set viewing angle
        self.ax.view_init(elev=20, azim=45)

    def _state_to_bloch(self, state: qt.Qobj) -> np.ndarray:
        """Convert quantum state to Bloch vector."""
        if state.type != "ket":
            state = state.dag()

        # Pauli matrices
        sx = qt.sigmax()
        sy = qt.sigmay()
        sz = qt.sigmaz()

        # Compute expectation values
        x = qt.expect(sx, state)
        y = qt.expect(sy, state)
        z = qt.expect(sz, state)

        return np.array([x, y, z])

    def show(self):
        """Display the animation."""
        if self.animation is None:
            raise ValueError("Must call create_animation() before show()")
        plt.show()

    def close(self):
        """Close the animation figure."""
        if self.fig is not None:
            plt.close(self.fig)


def create_bloch_animation(
    states: Union[List[qt.Qobj], List[List[qt.Qobj]]],
    labels: Optional[List[str]] = None,
    filename: Optional[str] = None,
    fps: int = 20,
    interval: int = 50,
    trail_length: Optional[int] = None,
    style: Optional[AnimationStyle] = None,
    **kwargs,
) -> BlochAnimator:
    """
    Convenience function to create and optionally save Bloch animation.

    Parameters
    ----------
    states : list of Qobj or list of list of Qobj
        Quantum states to animate
    labels : list of str, optional
        Labels for trajectories
    filename : str, optional
        If provided, save animation to this file
    fps : int
        Frames per second for saved animation
    interval : int
        Milliseconds between frames
    trail_length : int, optional
        Number of previous points to show
    style : AnimationStyle, optional
        Styling options
    **kwargs
        Additional arguments passed to BlochAnimator

    Returns
    -------
    animator : BlochAnimator

    Examples
    --------
    >>> states = [qt.Qobj([[np.cos(t)], [np.sin(t)]]) for t in np.linspace(0, np.pi, 50)]
    >>> animator = create_bloch_animation(states, filename='evolution.gif')
    """
    animator = BlochAnimator(states, labels=labels, style=style, **kwargs)
    animator.create_animation(interval=interval, trail_length=trail_length)

    if filename:
        animator.save(filename, fps=fps)

    return animator


def save_animation(
    animator: BlochAnimator,
    filename: str,
    fps: int = 20,
    dpi: int = 100,
):
    """
    Save an existing BlochAnimator to file.

    Parameters
    ----------
    animator : BlochAnimator
        Animator instance to save
    filename : str
        Output filename
    fps : int
        Frames per second
    dpi : int
        Resolution

    Examples
    --------
    >>> animator = BlochAnimator(states)
    >>> animator.create_animation()
    >>> save_animation(animator, 'output.gif', fps=30)
    """
    animator.save(filename, fps=fps, dpi=dpi)


def animate_pulse_evolution(
    hamiltonian: Callable,
    control_pulse: np.ndarray,
    times: np.ndarray,
    initial_state: qt.Qobj,
    **animation_kwargs,
) -> BlochAnimator:
    """
    Create animation of state evolution under a control pulse.

    Parameters
    ----------
    hamiltonian : callable
        Function H(t, args) returning Hamiltonian at time t
    control_pulse : ndarray
        Control amplitude values
    times : ndarray
        Time points
    initial_state : Qobj
        Initial quantum state
    **animation_kwargs
        Additional arguments for BlochAnimator

    Returns
    -------
    animator : BlochAnimator

    Examples
    --------
    >>> def H(t, args):
    ...     return args['H0'] + args['u'](t) * args['H1']
    >>> pulse = np.sin(np.linspace(0, 2*np.pi, 100))
    >>> times = np.linspace(0, 10, 100)
    >>> animator = animate_pulse_evolution(H, pulse, times, qt.basis(2, 0))
    """
    # Create time-dependent Hamiltonian
    from scipy.interpolate import interp1d

    pulse_func = interp1d(times, control_pulse, kind="linear", fill_value="extrapolate")

    # Build QuTiP format Hamiltonian
    H0 = qt.sigmaz() / 2  # Drift Hamiltonian (example)
    H1 = qt.sigmax() / 2  # Control Hamiltonian (example)

    H = [H0, [H1, lambda t, args: pulse_func(t)]]

    # Solve Schrödinger equation
    result = qt.mesolve(H, initial_state, times, [], [])

    # Create animation
    animator = BlochAnimator(result.states, **animation_kwargs)
    animator.create_animation()

    return animator
