"""
Configuration management for QubitPulseOpt.

This module provides a flexible configuration system supporting:
- YAML configuration files
- Environment variable overrides
- Programmatic configuration updates
- Configuration validation
- Default fallback values

Usage:
    >>> from src.config import Config, load_config
    >>>
    >>> # Load default configuration
    >>> config = load_config()
    >>>
    >>> # Load custom configuration
    >>> config = load_config("my_config.yaml")
    >>>
    >>> # Access configuration values
    >>> T1 = config.get("system.decoherence.T1")
    >>> max_iter = config.get("optimization.grape.max_iterations")
    >>>
    >>> # Override values programmatically
    >>> config.set("system.decoherence.T1", 100e-6)
    >>>
    >>> # Save modified configuration
    >>> config.save("modified_config.yaml")

Author: QubitPulseOpt Team
Date: 2025-01-28
"""

import yaml
import os
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


class Config:
    """
    Configuration manager for quantum control experiments.

    Supports nested dictionary access with dot notation and
    provides validation and override mechanisms.

    Attributes:
        data (dict): Configuration data dictionary
        source (Path): Path to source configuration file
    """

    def __init__(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
        source: Optional[Path] = None,
    ):
        """
        Initialize configuration.

        Args:
            config_dict: Configuration dictionary
            source: Source file path (for reference)
        """
        self.data = config_dict or {}
        self.source = source
        self._defaults = None

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "system.decoherence.T1")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get("system.qubit.frequency")
            5000000000.0
            >>> config.get("nonexistent.key", default=42)
            42
        """
        keys = key.split(".")
        value = self.data

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "system.decoherence.T1")
            value: Value to set

        Example:
            >>> config.set("system.decoherence.T1", 100e-6)
        """
        keys = key.split(".")
        d = self.data

        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]

        # Set the final value
        d[keys[-1]] = value

    def update(self, updates: Dict[str, Any], prefix: str = "") -> None:
        """
        Update configuration with a dictionary of values.

        Args:
            updates: Dictionary of updates
            prefix: Optional prefix for all keys

        Example:
            >>> config.update({"T1": 100e-6, "T2": 200e-6}, prefix="system.decoherence")
        """
        for key, value in updates.items():
            full_key = f"{prefix}.{key}" if prefix else key
            self.set(full_key, value)

    def merge(self, other: Union["Config", Dict[str, Any]]) -> None:
        """
        Merge another configuration into this one.

        Args:
            other: Another Config object or dictionary to merge
        """
        if isinstance(other, Config):
            other_data = other.data
        else:
            other_data = other

        self.data = self._deep_merge(self.data, other_data)

    def _deep_merge(self, base: Dict, updates: Dict) -> Dict:
        """
        Recursively merge two dictionaries.

        Args:
            base: Base dictionary
            updates: Updates to apply

        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(base)

        for key, value in updates.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return copy.deepcopy(self.data)

    def save(self, filepath: Union[str, Path]) -> Path:
        """
        Save configuration to YAML file.

        Args:
            filepath: Output file path

        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)

        return filepath

    def validate(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate configuration against a schema.

        Args:
            schema: Validation schema (optional)

        Returns:
            True if valid, raises ConfigError otherwise
        """
        # Basic validation - check required fields
        required_keys = [
            "system.qubit.frequency",
            "system.decoherence.T1",
            "system.decoherence.T2",
            "pulse.default.duration",
        ]

        for key in required_keys:
            if self.get(key) is None:
                raise ConfigError(f"Required configuration key missing: {key}")

        # Validate value ranges
        if self.get("system.decoherence.T1", 0) <= 0:
            raise ConfigError("T1 must be positive")

        if self.get("system.decoherence.T2", 0) <= 0:
            raise ConfigError("T2 must be positive")

        # T2 cannot exceed 2*T1 (physics constraint)
        T1 = self.get("system.decoherence.T1", 1)
        T2 = self.get("system.decoherence.T2", 1)
        if T2 > 2 * T1:
            warnings.warn(
                f"T2 ({T2}) > 2*T1 ({2 * T1}): this violates physical constraints"
            )

        return True

    def apply_env_overrides(self, prefix: str = "QUBITPULSEOPT_") -> None:
        """
        Apply configuration overrides from environment variables.

        Environment variables should be named like:
        QUBITPULSEOPT_SYSTEM__DECOHERENCE__T1=50e-6

        Args:
            prefix: Environment variable prefix
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert __ to .
                config_key = key[len(prefix) :].lower().replace("__", ".")

                # Try to parse value as number
                try:
                    if "e" in value.lower():
                        parsed_value = float(value)
                    elif "." in value:
                        parsed_value = float(value)
                    else:
                        parsed_value = int(value)
                except ValueError:
                    # Keep as string
                    parsed_value = value

                self.set(config_key, parsed_value)

    def __repr__(self) -> str:
        """String representation."""
        source_str = f" from {self.source}" if self.source else ""
        return f"Config({len(self.data)} sections{source_str})"

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting."""
        self.set(key, value)


def load_config(
    filepath: Optional[Union[str, Path]] = None,
    apply_env: bool = True,
    validate: bool = True,
) -> Config:
    """
    Load configuration from YAML file.

    Args:
        filepath: Path to configuration file. If None, loads default config.
        apply_env: Whether to apply environment variable overrides
        validate: Whether to validate the configuration

    Returns:
        Config object

    Example:
        >>> config = load_config()  # Load default
        >>> config = load_config("custom.yaml")  # Load custom
    """
    # Determine config file path
    if filepath is None:
        # Use default config
        default_config_path = (
            Path(__file__).parent.parent / "config" / "default_config.yaml"
        )
        filepath = default_config_path
    else:
        filepath = Path(filepath)

    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    # Load YAML
    with open(filepath, "r") as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        config_data = {}

    # Create Config object
    config = Config(config_data, source=filepath)

    # Apply environment overrides
    if apply_env:
        config.apply_env_overrides()

    # Validate
    if validate:
        try:
            config.validate()
        except ConfigError as e:
            warnings.warn(f"Configuration validation failed: {e}")

    return config


def create_config_from_dict(
    config_dict: Dict[str, Any], validate: bool = True
) -> Config:
    """
    Create Config object from dictionary.

    Args:
        config_dict: Configuration dictionary
        validate: Whether to validate the configuration

    Returns:
        Config object
    """
    config = Config(config_dict)

    if validate:
        try:
            config.validate()
        except ConfigError as e:
            warnings.warn(f"Configuration validation failed: {e}")

    return config


def merge_configs(base: Config, *overrides: Union[Config, Dict[str, Any]]) -> Config:
    """
    Merge multiple configurations together.

    Args:
        base: Base configuration
        *overrides: Additional configurations to merge (in order)

    Returns:
        Merged configuration

    Example:
        >>> default_config = load_config()
        >>> custom_overrides = {"system": {"decoherence": {"T1": 100e-6}}}
        >>> final_config = merge_configs(default_config, custom_overrides)
    """
    result = Config(base.to_dict(), source=base.source)

    for override in overrides:
        result.merge(override)

    return result


def get_default_config() -> Config:
    """
    Get the default configuration.

    Returns:
        Default Config object
    """
    return load_config(filepath=None, apply_env=False, validate=False)


# Convenience function for quick access
def get_config_value(
    key: str, config: Optional[Config] = None, default: Any = None
) -> Any:
    """
    Quick access to configuration value.

    Args:
        key: Configuration key
        config: Config object (loads default if None)
        default: Default value

    Returns:
        Configuration value
    """
    if config is None:
        config = load_config()

    return config.get(key, default)
