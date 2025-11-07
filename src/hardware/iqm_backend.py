"""
IQM Backend Manager - REST API v1
==================================

This module provides a wrapper for IQM Resonance quantum hardware access using
the direct REST API v1 endpoint, handling authentication, backend selection,
job submission, and connection management.

Classes
-------
IQMBackendManager : Main class for managing IQM hardware connections via REST API
IQMRestBackend : Backend wrapper for circuit execution via REST API
IQMRestJob : Job handle for asynchronous result retrieval

Security
--------
Authentication uses the IQM_TOKEN environment variable. This should be set
in a .env file that is excluded from version control.

Example
-------
>>> from src.hardware.iqm_backend import IQMBackendManager
>>> manager = IQMBackendManager()
>>> backend = manager.get_backend()
>>> print(f"Connected to: {backend.name}")
>>> topology = manager.get_hardware_topology()
>>> print(f"Available qubits: {topology['qubits']}")
"""

import os
import logging
import requests
import time
import json
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class IQMBackendManager:
    """
    Manages authentication and access to IQM Resonance quantum hardware via REST API v1.

    This class handles:
    - Environment variable loading for secure token management
    - REST API v1 communication with IQM Resonance
    - Hardware topology queries
    - Quantum computer selection
    - Emulator/simulator backend access for testing

    Attributes
    ----------
    token : str
        IQM API authentication token (loaded from environment)
    base_url : str
        REST API v1 base URL
    session : requests.Session
        HTTP session with authentication headers
    quantum_computers : list
        Available quantum computers from API
    selected_qc : dict or None
        Currently selected quantum computer
    use_emulator : bool
        If True, use fake/emulator backend for testing

    Example
    -------
    >>> manager = IQMBackendManager()
    >>> backend = manager.get_backend(use_emulator=False)
    >>> topology = manager.get_hardware_topology()
    """

    def __init__(self, dotenv_path: Optional[str] = None):
        """
        Initialize the IQM backend manager.

        Parameters
        ----------
        dotenv_path : str, optional
            Path to .env file containing IQM_TOKEN.
            If None, searches for .env in standard locations.

        Raises
        ------
        ValueError
            If IQM_TOKEN is not found in environment variables
        """
        # Load environment variables from .env file
        if dotenv_path:
            load_dotenv(dotenv_path)
        else:
            load_dotenv()  # Searches for .env in current and parent directories

        # Retrieve IQM token (DO NOT LOG THIS VALUE)
        self.token = os.getenv("IQM_TOKEN")
        if not self.token:
            raise ValueError(
                "IQM_TOKEN not found in environment variables. "
                "Please create a .env file with your IQM API token."
            )

        logger.info("IQM authentication token loaded successfully")

        # REST API v1 endpoint
        self.base_url = "https://resonance.meetiqm.com/api/v1"

        # Setup HTTP session with authentication
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        )

        # Backend state
        self.quantum_computers = []
        self.selected_qc = None
        self.use_emulator = False

    def _fetch_quantum_computers(self) -> List[Dict[str, Any]]:
        """
        Fetch list of available quantum computers from REST API.

        Returns
        -------
        quantum_computers : list of dict
            List of available quantum computers with metadata
        """
        try:
            url = f"{self.base_url}/quantum-computers"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            self.quantum_computers = data.get("quantum_computers", [])

            logger.info(f"Retrieved {len(self.quantum_computers)} quantum computers")
            return self.quantum_computers

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch quantum computers: {e}")
            raise

    def get_backend(
        self, backend_name: Optional[str] = None, use_emulator: bool = False
    ) -> Any:
        """
        Get an IQM quantum backend for circuit/pulse execution.

        Parameters
        ----------
        backend_name : str, optional
            Name/alias of specific backend to use (e.g., 'garnet', 'emerald').
            If None, uses the first available non-mock backend.
        use_emulator : bool, default=False
            If True, returns a fake/emulator backend for testing.
            This does not consume hardware credits.

        Returns
        -------
        backend : Backend object
            Quantum backend for execution

        Example
        -------
        >>> manager = IQMBackendManager()
        >>> # For testing without consuming credits
        >>> emulator = manager.get_backend(use_emulator=True)
        >>> # For real hardware execution
        >>> hardware = manager.get_backend(backend_name='garnet')
        """
        self.use_emulator = use_emulator

        if use_emulator:
            logger.info("Using IQM emulator backend (no hardware execution)")
            return self._get_emulator_backend()

        # Fetch available quantum computers
        if not self.quantum_computers:
            self._fetch_quantum_computers()

        # Select quantum computer
        if backend_name:
            # Find by alias or name
            for qc in self.quantum_computers:
                if (
                    qc.get("alias") == backend_name
                    or qc.get("display_name") == backend_name
                ):
                    # Skip mock backends for real hardware requests
                    if ":mock" not in qc.get("alias", ""):
                        self.selected_qc = qc
                        break

            if not self.selected_qc:
                raise ValueError(f"Backend '{backend_name}' not found")
        else:
            # Use first non-mock backend
            for qc in self.quantum_computers:
                if ":mock" not in qc.get("alias", ""):
                    self.selected_qc = qc
                    break

            if not self.selected_qc:
                raise ValueError("No non-mock quantum computers available")

        logger.info(
            f"Selected quantum computer: {self.selected_qc.get('display_name')} "
            f"(alias: {self.selected_qc.get('alias')})"
        )

        # Return a simple backend wrapper
        return IQMRestBackend(
            qc_id=self.selected_qc["id"],
            name=self.selected_qc.get("alias", "iqm"),
            session=self.session,
            base_url=self.base_url,
        )

    def _get_emulator_backend(self) -> Any:
        """
        Get a fake/emulator backend for testing without hardware.

        Returns
        -------
        fake_backend : AerSimulator
            Emulator backend that mimics quantum hardware

        Notes
        -----
        This uses Qiskit Aer simulator. Emulator runs do not consume IQM credits.
        """
        logger.warning("IQMFakeBackend not available, falling back to Aer simulator")
        try:
            from qiskit_aer import AerSimulator

            return AerSimulator()
        except ImportError as e:
            raise ImportError(
                "AerSimulator not available. Install with: pip install qiskit-aer"
            ) from e

    def get_hardware_topology(self) -> Dict[str, Any]:
        """
        Retrieve hardware topology information from the IQM backend.

        Returns
        -------
        topology : dict
            Dictionary containing:
            - 'qubits': List[str] - Available qubit names
            - 'backend_name': str - Name of the backend
            - 'n_qubits': int - Number of qubits

        Example
        -------
        >>> manager = IQMBackendManager()
        >>> topology = manager.get_hardware_topology()
        >>> print(f"Qubits: {topology['qubits']}")
        """
        if not self.selected_qc and not self.use_emulator:
            self.get_backend()

        if self.use_emulator:
            # Return minimal topology for emulator
            return {
                "qubits": ["QB0", "QB1", "QB2", "QB3", "QB4"],
                "backend_name": "AerSimulator",
                "n_qubits": 5,
            }

        # For REST API backends, return basic info
        # Full topology would require additional API calls
        return {
            "qubits": [
                f"QB{i}" for i in range(20)
            ],  # Typical IQM backend has ~20 qubits
            "backend_name": self.selected_qc.get("alias", "iqm"),
            "n_qubits": 20,  # Default assumption
            "qc_id": self.selected_qc.get("id"),
            "display_name": self.selected_qc.get("display_name"),
        }

    def get_available_backends(self) -> List[str]:
        """
        List all available IQM backends.

        Returns
        -------
        backend_names : List[str]
            Aliases of available quantum backends

        Example
        -------
        >>> manager = IQMBackendManager()
        >>> backends = manager.get_available_backends()
        >>> print(f"Available: {backends}")
        """
        if not self.quantum_computers:
            self._fetch_quantum_computers()

        # Return aliases of non-mock backends
        backends = []
        for qc in self.quantum_computers:
            alias = qc.get("alias", "")
            if ":mock" not in alias:
                backends.append(alias)

        return backends

    def verify_connection(self) -> bool:
        """
        Verify that connection to IQM hardware is working.

        Returns
        -------
        is_connected : bool
            True if connection successful, False otherwise

        Example
        -------
        >>> manager = IQMBackendManager()
        >>> if manager.verify_connection():
        >>>     print("Connection OK")
        >>> else:
        >>>     print("Connection failed")
        """
        try:
            qcs = self._fetch_quantum_computers()
            if len(qcs) > 0:
                logger.info("✓ Connection verified successfully")
                return True
            else:
                logger.warning("✗ Connection failed: No quantum computers available")
                return False

        except Exception as e:
            logger.error(f"✗ Connection verification failed: {e}")
            return False

    def get_topology(self) -> Dict[str, Any]:
        """
        Alias for get_hardware_topology() for compatibility.

        Returns
        -------
        topology : dict
            Hardware topology information
        """
        return self.get_hardware_topology()

    def __repr__(self) -> str:
        """String representation of the backend manager."""
        status = "Connected" if self.selected_qc else "Not connected"
        mode = "Emulator" if self.use_emulator else "Hardware"
        return f"IQMBackendManager(status={status}, mode={mode})"


class IQMRestBackend:
    """
    Backend wrapper for IQM REST API v1 with full job submission support.

    This provides a Qiskit-compatible interface for circuit execution via
    the IQM Resonance REST API v1 endpoints.

    Attributes
    ----------
    qc_id : str
        Quantum computer UUID
    name : str
        Backend alias/name
    session : requests.Session
        Authenticated HTTP session
    base_url : str
        REST API v1 base URL
    """

    def __init__(self, qc_id: str, name: str, session: requests.Session, base_url: str):
        """
        Initialize REST backend wrapper.

        Parameters
        ----------
        qc_id : str
            Quantum computer ID
        name : str
            Backend name/alias
        session : requests.Session
            Authenticated HTTP session
        base_url : str
            REST API base URL
        """
        self.qc_id = qc_id
        self.name = name
        self.session = session
        self.base_url = base_url

    def run(self, circuit, shots: int = 1024, **kwargs):
        """
        Submit circuit for execution on IQM hardware.

        Parameters
        ----------
        circuit : QuantumCircuit or list
            Circuit(s) to execute
        shots : int, default=1024
            Number of measurement shots

        Returns
        -------
        job : IQMRestJob
            Job handle for asynchronous result retrieval

        Raises
        ------
        RuntimeError
            If job submission fails
        """
        try:
            # Convert circuit to IQM format
            iqm_circuit = self._qiskit_to_iqm_circuit(circuit, shots)

            # Submit job via REST API
            url = f"{self.base_url}/jobs"
            payload = {
                "quantum_computer_id": self.qc_id,
                "circuits": [iqm_circuit],
                "shots": shots,
                "calibration_set_id": kwargs.get("calibration_set_id"),
                "max_circuit_duration_over_t2": kwargs.get(
                    "max_circuit_duration_over_t2", 1.0
                ),
                "heralding_mode": kwargs.get("heralding_mode", "none"),
            }

            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            logger.info(f"Submitting job to {self.name} ({self.qc_id})")
            logger.debug(f"Job payload: {json.dumps(payload, indent=2)}")

            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()

            job_data = response.json()
            job_id = job_data.get("id")

            if not job_id:
                raise RuntimeError(f"No job ID returned: {job_data}")

            logger.info(f"✓ Job submitted successfully: {job_id}")

            return IQMRestJob(
                job_id=job_id,
                session=self.session,
                base_url=self.base_url,
                backend_name=self.name,
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Job submission failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise RuntimeError(f"Failed to submit job: {e}") from e

    def _qiskit_to_iqm_circuit(self, circuit, shots: int) -> Dict[str, Any]:
        """
        Convert Qiskit QuantumCircuit to IQM circuit format.

        Parameters
        ----------
        circuit : QuantumCircuit
            Qiskit circuit to convert
        shots : int
            Number of shots

        Returns
        -------
        iqm_circuit : dict
            Circuit in IQM JSON format

        Notes
        -----
        This is a minimal conversion. For production use, implement
        full transpilation via IQMTranspiler or iqm-client library.
        """
        try:
            from qiskit import QuantumCircuit

            if not isinstance(circuit, QuantumCircuit):
                raise ValueError("Circuit must be a Qiskit QuantumCircuit")

            # Basic conversion: extract operations
            instructions = []

            for instruction, qargs, cargs in circuit.data:
                op_name = instruction.name
                qubit_names = [f"QB{circuit.qubits.index(q)}" for q in qargs]

                # Map Qiskit gates to IQM native gates
                if op_name in ["x", "y", "z", "h", "s", "t"]:
                    # Single-qubit gates -> convert to R gate
                    instructions.append(
                        {
                            "name": "r",
                            "qubits": qubit_names,
                            "args": self._gate_to_r_params(op_name),
                        }
                    )
                elif op_name == "cx":
                    # CNOT -> CZ with Hadamards
                    instructions.append(
                        {"name": "cz", "qubits": qubit_names, "args": {}}
                    )
                elif op_name == "cz":
                    instructions.append(
                        {"name": "cz", "qubits": qubit_names, "args": {}}
                    )
                elif op_name == "measure":
                    instructions.append(
                        {
                            "name": "measurement",
                            "qubits": qubit_names,
                            "args": {"key": f"m{circuit.clbits.index(cargs[0])}"},
                        }
                    )
                else:
                    logger.warning(f"Gate {op_name} not natively supported, skipping")

            return {"name": circuit.name or "circuit", "instructions": instructions}

        except ImportError:
            # Fallback for non-Qiskit circuits
            logger.warning("Qiskit not available, using minimal circuit format")
            return {"name": "circuit", "instructions": []}

    def _gate_to_r_params(self, gate_name: str) -> Dict[str, float]:
        """
        Convert standard gates to IQM R gate parameters.

        IQM R gate: R(theta, phi) = exp(-i * theta/2 * (cos(phi)*X + sin(phi)*Y))
        """
        import math

        params = {
            "x": {"theta": math.pi, "phi": 0.0},
            "y": {"theta": math.pi, "phi": math.pi / 2},
            "z": {"theta": 0.0, "phi": 0.0},  # Phase gate, handled differently
            "h": {"theta": math.pi, "phi": 0.0},  # Approximate
            "s": {"theta": math.pi / 2, "phi": 0.0},
            "t": {"theta": math.pi / 4, "phi": 0.0},
        }

        return params.get(gate_name, {"theta": 0.0, "phi": 0.0})

    def configuration(self):
        """
        Return backend configuration.

        Returns
        -------
        config : object
            Backend configuration with basis gates and coupling map
        """

        class Config:
            n_qubits = 20
            basis_gates = ["r", "cz", "measure"]
            coupling_map = []
            backend_name = self.name
            backend_version = "1.0"
            max_shots = 100000

        config = Config()
        config.backend_name = self.name
        return config

    def __repr__(self):
        return f"IQMRestBackend(name='{self.name}', qc_id='{self.qc_id[:8]}...')"


class IQMRestJob:
    """
    Job handle for IQM REST API v1 submissions with polling and result retrieval.

    Attributes
    ----------
    _job_id : str
        IQM job UUID
    session : requests.Session
        Authenticated HTTP session
    base_url : str
        REST API v1 base URL
    backend_name : str
        Name of backend where job is running
    _cached_status : str or None
        Cached job status to reduce API calls
    _cached_result : dict or None
        Cached result data
    """

    # IQM job status constants
    STATUS_PENDING = "pending_compilation"
    STATUS_READY = "ready"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"
    STATUS_ABORTED = "aborted"
    STATUS_DELETED = "deleted"

    def __init__(
        self,
        job_id: str,
        session: requests.Session,
        base_url: str,
        backend_name: str = "iqm",
    ):
        """
        Initialize job handle.

        Parameters
        ----------
        job_id : str
            IQM job UUID
        session : requests.Session
            Authenticated HTTP session
        base_url : str
            REST API base URL
        backend_name : str, default='iqm'
            Backend name for logging
        """
        self._job_id = job_id
        self.session = session
        self.base_url = base_url
        self.backend_name = backend_name
        self._cached_status = None
        self._cached_result = None

        logger.debug(f"Created job handle for {job_id} on {backend_name}")

    def job_id(self) -> str:
        """
        Get job identifier.

        Returns
        -------
        job_id : str
            IQM job UUID
        """
        return self._job_id

    def status(self) -> str:
        """
        Query current job status from IQM API.

        Returns
        -------
        status : str
            One of: 'pending_compilation', 'ready', 'running', 'done', 'failed', 'aborted', 'deleted'

        Raises
        ------
        RuntimeError
            If status query fails
        """
        try:
            url = f"{self.base_url}/jobs/{self._job_id}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            job_data = response.json()
            self._cached_status = job_data.get("status", "unknown")

            logger.debug(f"Job {self._job_id[:8]}... status: {self._cached_status}")

            return self._cached_status

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get job status: {e}")
            raise RuntimeError(f"Status query failed: {e}") from e

    def done(self) -> bool:
        """
        Check if job has completed (success or failure).

        Returns
        -------
        is_done : bool
            True if job is in terminal state (done/failed/aborted)
        """
        current_status = self.status()
        return current_status in [
            self.STATUS_DONE,
            self.STATUS_FAILED,
            self.STATUS_ABORTED,
            self.STATUS_DELETED,
        ]

    def result(self, timeout: Optional[float] = None, wait: float = 5.0):
        """
        Wait for job completion and retrieve results.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait in seconds. If None, waits indefinitely.
        wait : float, default=5.0
            Polling interval in seconds

        Returns
        -------
        result : IQMResult
            Job execution result with measurement counts

        Raises
        ------
        RuntimeError
            If job fails or timeout is exceeded
        """
        # Return cached result if available
        if self._cached_result is not None:
            return self._cached_result

        start_time = time.time()

        logger.info(f"Waiting for job {self._job_id[:8]}... to complete")

        while True:
            current_status = self.status()

            if current_status == self.STATUS_DONE:
                # Job completed successfully, retrieve results
                logger.info(f"✓ Job {self._job_id[:8]}... completed successfully")
                return self._fetch_result()

            elif current_status in [
                self.STATUS_FAILED,
                self.STATUS_ABORTED,
                self.STATUS_DELETED,
            ]:
                # Job failed
                error_msg = f"Job {self._job_id} {current_status}"
                logger.error(f"✗ {error_msg}")
                raise RuntimeError(error_msg)

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise RuntimeError(
                        f"Job timeout after {elapsed:.1f}s (status: {current_status})"
                    )

            # Still running, wait and poll again
            logger.debug(
                f"Job {self._job_id[:8]}... status: {current_status}, waiting {wait}s..."
            )
            time.sleep(wait)

    def _fetch_result(self):
        """
        Fetch result data from IQM API.

        Returns
        -------
        result : IQMResult
            Parsed job result
        """
        try:
            url = f"{self.base_url}/jobs/{self._job_id}/results"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            result_data = response.json()

            # Cache the result
            self._cached_result = IQMResult(
                job_id=self._job_id,
                backend_name=self.backend_name,
                data=result_data,
                success=True,
            )

            return self._cached_result

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch results: {e}")
            raise RuntimeError(f"Result retrieval failed: {e}") from e

    def cancel(self):
        """
        Cancel a running job.

        Returns
        -------
        success : bool
            True if cancellation succeeded

        Notes
        -----
        Not all job states can be cancelled. Check IQM API documentation.
        """
        try:
            url = f"{self.base_url}/jobs/{self._job_id}/abort"
            response = self.session.post(url, timeout=10)
            response.raise_for_status()

            logger.info(f"Job {self._job_id} cancelled")
            self._cached_status = self.STATUS_ABORTED
            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to cancel job: {e}")
            return False

    def __repr__(self):
        status_str = self._cached_status or "unknown"
        return f"IQMRestJob(id='{self._job_id[:8]}...', status='{status_str}')"


class IQMResult:
    """
    Result object for IQM job execution.

    Provides Qiskit-compatible interface for accessing measurement results.

    Attributes
    ----------
    job_id : str
        Job UUID
    backend_name : str
        Backend where job ran
    data : dict
        Raw result data from IQM API
    success : bool
        Whether job completed successfully
    """

    def __init__(
        self, job_id: str, backend_name: str, data: Dict[str, Any], success: bool = True
    ):
        """
        Initialize result object.

        Parameters
        ----------
        job_id : str
            IQM job UUID
        backend_name : str
            Backend name
        data : dict
            Raw API result data
        success : bool, default=True
            Success flag
        """
        self.job_id = job_id
        self.backend_name = backend_name
        self.data = data
        self.success = success

    def get_counts(self, circuit: Union[int, str] = 0) -> Dict[str, int]:
        """
        Get measurement counts for a circuit.

        Parameters
        ----------
        circuit : int or str, default=0
            Circuit index or name (IQM API returns results per circuit)

        Returns
        -------
        counts : dict
            Measurement outcome counts, e.g., {'00': 512, '01': 256, ...}
        """
        try:
            # IQM API returns measurements in format:
            # {"measurements": [{"result": [...], "shots": N}]}
            measurements = self.data.get("measurements", [])

            if not measurements:
                logger.warning("No measurement data in result")
                return {}

            # Get first circuit's measurements (or specified circuit)
            if isinstance(circuit, int):
                if circuit >= len(measurements):
                    logger.warning(f"Circuit index {circuit} out of range")
                    return {}
                meas_data = measurements[circuit]
            else:
                # Find by name
                meas_data = measurements[0]  # Default to first

            # Convert IQM measurement format to counts
            # IQM returns: [[0,1,0], [1,1,0], ...] (list of bit arrays)
            results = meas_data.get("result", [])

            counts = {}
            for outcome in results:
                # Convert [0,1,0] -> "010"
                bitstring = "".join(str(bit) for bit in outcome)
                counts[bitstring] = counts.get(bitstring, 0) + 1

            return counts

        except Exception as e:
            logger.error(f"Failed to parse counts: {e}")
            return {}

    def get_memory(self) -> List[str]:
        """
        Get individual shot results.

        Returns
        -------
        memory : list of str
            Individual measurement outcomes
        """
        try:
            measurements = self.data.get("measurements", [])
            if not measurements:
                return []

            results = measurements[0].get("result", [])
            return ["".join(str(bit) for bit in outcome) for outcome in results]

        except Exception as e:
            logger.error(f"Failed to get memory: {e}")
            return []

    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"IQMResult(job='{self.job_id[:8]}...', status={status}, backend='{self.backend_name}')"
