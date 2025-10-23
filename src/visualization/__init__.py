"""
Visualization and interactive tools for quantum control.

This package provides:
- Interactive dashboards for optimization monitoring
- Bloch sphere animations
- Report generation and publication-quality figures
"""

from .dashboard import (
    OptimizationDashboard,
    ParameterSweepViewer,
    PulseComparisonViewer,
    BlochViewer3D,
)

from .bloch_animation import (
    BlochAnimator,
    create_bloch_animation,
    save_animation,
)

from .reports import (
    PulseReport,
    OptimizationReport,
    generate_latex_table,
    create_publication_figure,
)

__all__ = [
    # Dashboard
    "OptimizationDashboard",
    "ParameterSweepViewer",
    "PulseComparisonViewer",
    "BlochViewer3D",
    # Animation
    "BlochAnimator",
    "create_bloch_animation",
    "save_animation",
    # Reports
    "PulseReport",
    "OptimizationReport",
    "generate_latex_table",
    "create_publication_figure",
]
