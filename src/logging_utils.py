"""
Logging and diagnostics utilities for QubitPulseOpt.

This module provides structured logging with:
- Configurable log levels and formats
- File and console handlers
- Context managers for operation tracking
- Performance profiling utilities
- Debug diagnostics helpers

Usage:
    >>> from src.logging_utils import get_logger, log_operation
    >>>
    >>> # Get a logger
    >>> logger = get_logger("optimization")
    >>>
    >>> # Log with context
    >>> with log_operation("GRAPE optimization", logger):
    ...     result = run_grape_optimization()
    >>>
    >>> # Profile performance
    >>> from src.logging_utils import profile_function
    >>>
    >>> @profile_function
    ... def expensive_operation():
    ...     # Do work
    ...     pass

Author: QubitPulseOpt Team
Date: 2025-01-28
"""

import logging
import sys
import time
import functools
from pathlib import Path
from typing import Optional, Callable, Any, Dict
from contextlib import contextmanager
import traceback
import json
from datetime import datetime


# Global logger registry
_LOGGERS: Dict[str, logging.Logger] = {}

# Default logging configuration
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = logging.INFO


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that can output structured JSON logs.
    """

    def __init__(self, fmt=None, datefmt=None, json_output=False):
        super().__init__(fmt, datefmt)
        self.json_output = json_output

    def format(self, record):
        if self.json_output:
            log_data = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

            # Add extra fields if present
            if hasattr(record, "extra"):
                log_data.update(record.extra)

            return json.dumps(log_data)
        else:
            return super().format(record)


def setup_logging(
    level: int = DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    json_output: bool = False,
    console: bool = True,
) -> None:
    """
    Setup global logging configuration.

    Args:
        level: Logging level (e.g., logging.INFO)
        log_file: Path to log file (None for no file logging)
        log_format: Log message format string
        date_format: Date format string
        json_output: Whether to output JSON-formatted logs
        console: Whether to log to console
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = StructuredFormatter(log_format, date_format, json_output)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(
    name: str, level: Optional[int] = None, create_if_missing: bool = True
) -> logging.Logger:
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name (typically module name)
        level: Optional logging level override
        create_if_missing: Whether to create logger if it doesn't exist

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger("optimization.grape")
        >>> logger.info("Starting GRAPE optimization")
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    if not create_if_missing:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    _LOGGERS[name] = logger
    return logger


@contextmanager
def log_operation(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    log_time: bool = True,
    log_errors: bool = True,
):
    """
    Context manager for logging operations with timing and error handling.

    Args:
        operation_name: Name of the operation being performed
        logger: Logger to use (creates default if None)
        level: Log level for start/end messages
        log_time: Whether to log execution time
        log_errors: Whether to log exceptions

    Example:
        >>> with log_operation("GRAPE optimization", logger):
        ...     result = optimize_pulse()
    """
    if logger is None:
        logger = get_logger(__name__)

    start_time = time.time()
    logger.log(level, f"Starting: {operation_name}")

    try:
        yield
        elapsed = time.time() - start_time

        if log_time:
            logger.log(level, f"Completed: {operation_name} (elapsed: {elapsed:.3f}s)")
        else:
            logger.log(level, f"Completed: {operation_name}")

    except Exception as e:
        elapsed = time.time() - start_time

        if log_errors:
            logger.error(
                f"Failed: {operation_name} after {elapsed:.3f}s - {type(e).__name__}: {e}"
            )
            logger.debug(f"Traceback:\n{traceback.format_exc()}")

        raise


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile function execution time.

    Args:
        func: Function to profile

    Returns:
        Wrapped function that logs execution time

    Example:
        >>> @profile_function
        ... def expensive_calculation():
        ...     time.sleep(1)
        ...     return 42
    """
    logger = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = f"{func.__module__}.{func.__name__}"

        logger.debug(f"Profiling: {func_name} started")

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Profiling: {func_name} completed in {elapsed:.4f}s")
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Profiling: {func_name} failed after {elapsed:.4f}s - {e}")
            raise

    return wrapper


class PerformanceTimer:
    """
    Context manager for timing code blocks.

    Example:
        >>> timer = PerformanceTimer("Matrix multiplication")
        >>> with timer:
        ...     result = A @ B
        >>> print(f"Elapsed: {timer.elapsed}s")
    """

    def __init__(
        self, name: str = "operation", logger: Optional[logging.Logger] = None
    ):
        self.name = name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Timer started: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        self.logger.debug(f"Timer stopped: {self.name} - {self.elapsed:.4f}s")


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """
    Log system and environment information.

    Args:
        logger: Logger to use
    """
    if logger is None:
        logger = get_logger(__name__)

    import platform
    import sys

    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Architecture: {platform.machine()}")

    try:
        import numpy as np

        logger.info(f"NumPy: {np.__version__}")
    except ImportError:
        pass

    try:
        import scipy

        logger.info(f"SciPy: {scipy.__version__}")
    except ImportError:
        pass

    try:
        import qutip

        logger.info(f"QuTiP: {qutip.__version__}")
    except ImportError:
        pass

    logger.info("=" * 60)


def log_config(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """
    Log configuration settings.

    Args:
        config: Configuration dictionary
        logger: Logger to use
    """
    assert isinstance(config, dict), f"Config must be a dictionary, got {type(config)}"

    if logger is None:
        logger = get_logger(__name__)

    logger.info("Configuration:")
    logger.info("-" * 60)

    # Rule 1: Replace recursion with iterative stack-based approach
    # Stack contains tuples of (dict, indent_level, prefix)
    stack = [(config, 0, "")]
    MAX_DEPTH = 10  # Rule 2: Explicit depth bound

    while stack:
        current_dict, indent, prefix = stack.pop()

        # Rule 5: Assertion for depth bound
        assert indent < MAX_DEPTH, (
            f"Config nesting depth {indent} exceeds maximum {MAX_DEPTH}"
        )

        # Process items in reverse order so they appear in correct order when popped
        items = list(current_dict.items())
        for key, value in reversed(items):
            if isinstance(value, dict):
                logger.info("  " * indent + f"{prefix}{key}:")
                # Push nested dict onto stack for processing
                stack.append((value, indent + 1, ""))
            else:
                logger.info("  " * indent + f"{prefix}{key}: {value}")

    logger.info("-" * 60)


class DiagnosticCollector:
    """
    Collect diagnostic information during operations.

    Example:
        >>> diag = DiagnosticCollector()
        >>> diag.record("fidelity", 0.995)
        >>> diag.record("iterations", 150)
        >>> diag.save("diagnostics.json")
    """

    def __init__(self, name: str = "diagnostics"):
        self.name = name
        self.data: Dict[str, Any] = {
            "name": name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "records": {},
            "events": [],
        }

    def record(self, key: str, value: Any) -> None:
        """
        Record a diagnostic value.

        Args:
            key: Diagnostic key
            value: Value to record
        """
        self.data["records"][key] = value

    def event(self, event_name: str, **kwargs) -> None:
        """
        Record an event with optional metadata.

        Args:
            event_name: Name of the event
            **kwargs: Additional event metadata
        """
        event_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event_name,
            **kwargs,
        }
        self.data["events"].append(event_data)

    def save(self, filepath: str) -> None:
        """
        Save diagnostics to JSON file.

        Args:
            filepath: Output file path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def get_summary(self) -> str:
        """
        Get a summary string of collected diagnostics.

        Returns:
            Summary string
        """
        summary = [f"Diagnostics: {self.name}"]
        summary.append(f"Timestamp: {self.data['timestamp']}")
        summary.append(f"Records: {len(self.data['records'])}")
        summary.append(f"Events: {len(self.data['events'])}")

        if self.data["records"]:
            summary.append("\nRecords:")
            for key, value in self.data["records"].items():
                summary.append(f"  {key}: {value}")

        return "\n".join(summary)


def debug_array_info(
    array, name: str = "array", logger: Optional[logging.Logger] = None
):
    """
    Log detailed information about a numpy array for debugging.

    Args:
        array: NumPy array to inspect
        name: Name for the array
        logger: Logger to use
    """
    if logger is None:
        logger = get_logger(__name__)

    import numpy as np

    logger.debug(f"Array '{name}':")
    logger.debug(f"  Shape: {array.shape}")
    logger.debug(f"  Dtype: {array.dtype}")
    logger.debug(f"  Min: {np.min(array)}")
    logger.debug(f"  Max: {np.max(array)}")
    logger.debug(f"  Mean: {np.mean(array)}")
    logger.debug(f"  Std: {np.std(array)}")

    if np.iscomplexobj(array):
        logger.debug(f"  Real range: [{np.min(array.real)}, {np.max(array.real)}]")
        logger.debug(f"  Imag range: [{np.min(array.imag)}, {np.max(array.imag)}]")

    # Check for special values
    if np.any(np.isnan(array)):
        logger.warning(f"  Contains NaN values: {np.sum(np.isnan(array))}")
    if np.any(np.isinf(array)):
        logger.warning(f"  Contains Inf values: {np.sum(np.isinf(array))}")


# Initialize default logging on import
setup_logging()
