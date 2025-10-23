# Visualization Package

Comprehensive visualization and interactive tools for quantum control optimization.

## Overview

This package provides a complete suite of visualization tools for monitoring, analyzing, and reporting quantum control optimization results. Features include real-time dashboards, Bloch sphere animations, parameter space exploration, and publication-quality figure generation.

## Modules

### `dashboard.py`
Interactive dashboards for optimization monitoring and analysis.

**Classes:**
- `OptimizationDashboard` - Real-time optimization monitoring
- `ParameterSweepViewer` - Parameter space exploration
- `PulseComparisonViewer` - Side-by-side pulse comparison
- `BlochViewer3D` - 3D Bloch sphere visualization

### `bloch_animation.py`
Bloch sphere animation tools for state evolution visualization.

**Classes:**
- `BlochAnimator` - Animation engine for quantum state trajectories
- `AnimationStyle` - Styling configuration dataclass

**Functions:**
- `create_bloch_animation()` - Quick animation creation
- `save_animation()` - Export animations
- `animate_pulse_evolution()` - Pulse-driven evolution

### `reports.py`
Analysis and reporting tools for pulse characterization.

**Classes:**
- `PulseReport` - Comprehensive pulse characterization
- `OptimizationReport` - Optimization process tracking
- `PulseCharacteristics` - Pulse metrics dataclass

**Functions:**
- `generate_latex_table()` - LaTeX table generation
- `create_publication_figure()` - Publication-quality figures

## Quick Start

### Monitor Optimization
```python
from src.visualization.dashboard import OptimizationDashboard

dashboard = OptimizationDashboard(n_controls=2, interactive=False)
for iteration in range(100):
    dashboard.update(
        iteration=iteration,
        fidelity=fidelity,
        gradient_norm=grad_norm,
        controls=control_array
    )
dashboard.save("optimization_progress.png")
```

### Create Bloch Animation
```python
from src.visualization.bloch_animation import create_bloch_animation

# states = list of quantum states
animator = create_bloch_animation(
    states,
    filename='evolution.gif',
    fps=20,
    trail_length=15
)
```

### Generate Pulse Report
```python
from src.visualization.reports import PulseReport

report = PulseReport(
    pulse,
    times=times,
    fidelity=0.995,
    target_gate='Hadamard'
)
report.generate_full_report('pulse_analysis.png')
report.export_metrics_table('metrics.tex', format='latex')
```

### Explore Parameter Space
```python
from src.visualization.dashboard import ParameterSweepViewer

viewer = ParameterSweepViewer()
fig, axes = viewer.plot_heatmap(
    x_values, y_values, z_values,
    x_label='Amplitude',
    y_label='Duration',
    z_label='Fidelity'
)
```

## Features

### Real-Time Dashboards
- Live fidelity and infidelity tracking
- Gradient norm visualization
- Control field evolution
- Custom metric support
- Interactive and batch modes

### Parameter Exploration
- 2D heatmaps with colorbars
- Contour plots
- Cross-section analysis
- 3D surface plots
- Optimal parameter identification

### Pulse Analysis
- Time-domain analysis
- Frequency-domain (FFT) analysis
- Comprehensive metrics:
  - Duration
  - Peak/RMS amplitude
  - Energy
  - Bandwidth
  - Smoothness
  - Fidelity
- Multi-pulse comparison

### Bloch Sphere
- 3D state visualization
- Trajectory plotting
- Multi-trajectory comparison
- Color-coded evolution
- Customizable rendering

### Animations
- GIF and MP4 export
- Adjustable frame rate and DPI
- Trail effects
- Multiple trajectories
- Custom styling

### Reporting
- Automated report generation
- Multiple export formats:
  - LaTeX tables
  - CSV data
  - JSON metadata
  - PNG/PDF figures
- Publication-ready quality

## Examples

See `examples/task4_demo.py` for comprehensive demonstrations of all features.

## Dependencies

### Required
- `numpy` - Numerical operations
- `matplotlib` - Plotting
- `qutip` - Quantum state manipulation
- `scipy` - Signal processing

### Optional
- `pillow` - GIF export (PillowWriter)
- `ffmpeg` - MP4 export (FFMpegWriter)

## API Reference

### OptimizationDashboard

```python
OptimizationDashboard(
    n_controls: int = 1,
    figsize: Tuple[int, int] = (14, 10),
    interactive: bool = True
)
```

**Methods:**
- `update(iteration, fidelity, gradient_norm, controls, **custom_metrics)` - Add iteration data
- `save(filename, dpi=300)` - Save dashboard figure
- `export_data()` - Export tracked data as dictionary
- `close()` - Close dashboard

### BlochAnimator

```python
BlochAnimator(
    trajectories: Union[List[Qobj], List[List[Qobj]]],
    labels: Optional[List[str]] = None,
    style: Optional[AnimationStyle] = None,
    figsize: Tuple[int, int] = (8, 8)
)
```

**Methods:**
- `create_animation(interval=50, trail_length=None, show_trail=True)` - Create animation
- `save(filename, fps=20, dpi=100)` - Export animation
- `show()` - Display animation
- `close()` - Close figure

### PulseReport

```python
PulseReport(
    pulse: np.ndarray,
    times: Optional[np.ndarray] = None,
    fidelity: Optional[float] = None,
    target_gate: Optional[str] = None,
    optimization_method: Optional[str] = None,
    label: str = "Main Pulse"
)
```

**Methods:**
- `add_comparison(pulse, label, metrics=None)` - Add comparison pulse
- `generate_full_report(filename=None)` - Generate visual report
- `export_metrics_table(filename, format='latex')` - Export metrics
- `characteristics` - Access computed pulse characteristics

### AnimationStyle

```python
AnimationStyle(
    sphere_alpha: float = 0.1,
    sphere_color: str = "gray",
    trajectory_linewidth: float = 2.0,
    point_size: int = 150,
    colormap: str = "viridis",
    background_color: str = "white"
)
```

## Testing

Run tests with pytest:
```bash
pytest tests/unit/test_dashboard.py -v        # 35 tests
pytest tests/unit/test_bloch_animation.py -v  # 31 tests
pytest tests/unit/test_reports.py -v          # 41 tests
```

Total: **107 passing tests**

## Performance Tips

1. Use `interactive=False` for batch processing
2. Reduce DPI during development (50-100 instead of 300)
3. Limit trail length in animations for memory efficiency
4. Close figures after saving to free memory
5. Use compressed formats for animations (GIF over MP4 for small files)

## Troubleshooting

### Animation export fails
- Install `pillow` for GIF: `pip install pillow`
- Install `ffmpeg` for MP4: `conda install ffmpeg` or system package manager

### Figure layout warnings
- Normal for complex multi-panel figures
- Use `bbox_inches='tight'` when saving

### Memory issues with animations
- Reduce number of frames
- Limit trail length
- Lower DPI and resolution
- Close figures promptly

## License

Part of the Quantum Controls Simulation Project.

## See Also

- `docs/TASK_4_SUMMARY.md` - Detailed implementation documentation
- `examples/task4_demo.py` - Comprehensive demonstrations
- `tests/unit/test_*.py` - Unit tests and usage examples