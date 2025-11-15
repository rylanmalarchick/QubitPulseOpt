"""
Unit Tests for IQM Backend Manager
===================================

Tests for the IQM hardware integration layer using mocked REST API responses.
All tests use mocked API calls - no real quantum hardware required.

Test Coverage
-------------
- IQMBackendManager initialization and authentication
- Backend selection and topology queries
- REST API communication (mocked)
- Job submission and status tracking
- Result retrieval and parsing
- Error handling and connection verification

Author: QubitPulseOpt Team
"""

import pytest
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import RequestException, HTTPError

from src.hardware.iqm_backend import (
    IQMBackendManager,
    IQMRestBackend,
    IQMRestJob,
    IQMResult,
)


# ============================================================================
# Fixtures for Mocked API Responses
# ============================================================================


@pytest.fixture
def mock_quantum_computers():
    """Mock response for quantum computers API endpoint."""
    return {
        "quantum_computers": [
            {
                "id": "qc-12345-real",
                "alias": "garnet",
                "display_name": "IQM Garnet",
                "status": "online",
            },
            {
                "id": "qc-67890-real",
                "alias": "emerald",
                "display_name": "IQM Emerald",
                "status": "online",
            },
            {
                "id": "qc-mock-123",
                "alias": "test:mock",
                "display_name": "Mock Backend",
                "status": "online",
            },
        ]
    }


@pytest.fixture
def mock_job_response():
    """Mock response for job submission."""
    return {
        "id": "job-abc123",
        "status": "pending_compilation",
        "created_at": "2025-11-15T12:00:00Z",
    }


@pytest.fixture
def mock_job_status_done():
    """Mock response for completed job status."""
    return {"id": "job-abc123", "status": "done", "completed_at": "2025-11-15T12:01:00Z"}


@pytest.fixture
def mock_job_result():
    """Mock response for job results."""
    return {
        "measurements": [
            {
                "result": [
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                ],
                "shots": 6,
            }
        ]
    }


@pytest.fixture
def mock_env_token(monkeypatch):
    """Set mock IQM_TOKEN environment variable."""
    monkeypatch.setenv("IQM_TOKEN", "test-token-12345")


# ============================================================================
# Test Class: IQMBackendManager Initialization
# ============================================================================


class TestIQMBackendManagerInit:
    """Tests for IQMBackendManager initialization and authentication."""

    def test_init_with_valid_token(self, mock_env_token):
        """Test manager initialization with valid token."""
        manager = IQMBackendManager()

        assert manager.token == "test-token-12345"
        assert manager.base_url == "https://resonance.meetiqm.com/api/v1"
        assert manager.session is not None
        assert "Authorization" in manager.session.headers
        assert manager.session.headers["Authorization"] == "Bearer test-token-12345"

    def test_init_without_token_raises_error(self, tmp_path):
        """Test that missing IQM_TOKEN raises ValueError."""
        # Save current token if exists
        import os
        old_token = os.environ.get("IQM_TOKEN")
        
        # Remove token
        if "IQM_TOKEN" in os.environ:
            del os.environ["IQM_TOKEN"]
        
        # Create empty .env file to prevent loading from actual .env
        env_file = tmp_path / ".env"
        env_file.write_text("")
        
        try:
            with pytest.raises(ValueError, match="IQM_TOKEN not found"):
                IQMBackendManager(dotenv_path=str(env_file))
        finally:
            # Restore old token
            if old_token is not None:
                os.environ["IQM_TOKEN"] = old_token

    def test_init_sets_correct_headers(self, mock_env_token):
        """Test that HTTP session has correct headers."""
        manager = IQMBackendManager()

        headers = manager.session.headers
        assert headers["Accept"] == "application/json"
        assert "Bearer" in str(headers["Authorization"])

    def test_init_state_defaults(self, mock_env_token):
        """Test that initial state is set correctly."""
        manager = IQMBackendManager()

        assert manager.quantum_computers == []
        assert manager.selected_qc is None
        assert manager.use_emulator is False


# ============================================================================
# Test Class: Backend Selection and Topology
# ============================================================================


class TestBackendSelection:
    """Tests for backend selection and hardware topology queries."""

    @patch("requests.Session.get")
    def test_fetch_quantum_computers(self, mock_get, mock_env_token, mock_quantum_computers):
        """Test fetching list of quantum computers."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = mock_quantum_computers
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        manager = IQMBackendManager()
        qcs = manager._fetch_quantum_computers()

        assert len(qcs) == 3
        assert qcs[0]["alias"] == "garnet"
        assert qcs[1]["alias"] == "emerald"
        mock_get.assert_called_once()

    @patch("requests.Session.get")
    def test_get_backend_by_name(self, mock_get, mock_env_token, mock_quantum_computers):
        """Test selecting backend by name."""
        mock_response = Mock()
        mock_response.json.return_value = mock_quantum_computers
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        manager = IQMBackendManager()
        backend = manager.get_backend(backend_name="emerald")

        assert backend.name == "emerald"
        assert backend.qc_id == "qc-67890-real"
        assert manager.selected_qc is not None
        assert manager.selected_qc["alias"] == "emerald"

    @patch("requests.Session.get")
    def test_get_backend_default_selection(self, mock_get, mock_env_token, mock_quantum_computers):
        """Test that default backend is first non-mock."""
        mock_response = Mock()
        mock_response.json.return_value = mock_quantum_computers
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        manager = IQMBackendManager()
        backend = manager.get_backend()

        # Should select 'garnet' (first non-mock)
        assert backend.name == "garnet"
        assert ":mock" not in backend.name

    @patch("requests.Session.get")
    def test_get_backend_skips_mock_backends(self, mock_get, mock_env_token, mock_quantum_computers):
        """Test that mock backends are skipped for real hardware."""
        mock_response = Mock()
        mock_response.json.return_value = mock_quantum_computers
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        manager = IQMBackendManager()
        backend = manager.get_backend()

        # Should not select 'test:mock'
        assert manager.selected_qc is not None
        assert ":mock" not in manager.selected_qc["alias"]

    def test_get_backend_emulator_mode(self, mock_env_token):
        """Test getting emulator backend."""
        manager = IQMBackendManager()

        with patch("qiskit_aer.AerSimulator") as mock_aer:
            mock_simulator = Mock()
            mock_aer.return_value = mock_simulator

            backend = manager.get_backend(use_emulator=True)

            assert backend == mock_simulator
            assert manager.use_emulator is True
            mock_aer.assert_called_once()

    @patch("requests.Session.get")
    def test_get_backend_invalid_name_raises_error(
        self, mock_get, mock_env_token, mock_quantum_computers
    ):
        """Test that invalid backend name raises ValueError."""
        mock_response = Mock()
        mock_response.json.return_value = mock_quantum_computers
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        manager = IQMBackendManager()

        with pytest.raises(ValueError, match="Backend 'nonexistent' not found"):
            manager.get_backend(backend_name="nonexistent")

    @patch("requests.Session.get")
    def test_get_hardware_topology(self, mock_get, mock_env_token, mock_quantum_computers):
        """Test getting hardware topology information."""
        mock_response = Mock()
        mock_response.json.return_value = mock_quantum_computers
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        manager = IQMBackendManager()
        manager.get_backend(backend_name="garnet")

        topology = manager.get_hardware_topology()

        assert "qubits" in topology
        assert "backend_name" in topology
        assert "n_qubits" in topology
        assert topology["backend_name"] == "garnet"
        assert topology["n_qubits"] == 20

    def test_get_topology_emulator(self, mock_env_token):
        """Test topology for emulator backend."""
        manager = IQMBackendManager()

        with patch("qiskit_aer.AerSimulator"):
            manager.get_backend(use_emulator=True)
            topology = manager.get_hardware_topology()

            assert topology["backend_name"] == "AerSimulator"
            assert topology["n_qubits"] == 5
            assert topology["qubits"] == ["QB0", "QB1", "QB2", "QB3", "QB4"]

    @patch("requests.Session.get")
    def test_get_available_backends(self, mock_get, mock_env_token, mock_quantum_computers):
        """Test listing available backends."""
        mock_response = Mock()
        mock_response.json.return_value = mock_quantum_computers
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        manager = IQMBackendManager()
        backends = manager.get_available_backends()

        assert "garnet" in backends
        assert "emerald" in backends
        assert "test:mock" not in backends  # Mock backends excluded


# ============================================================================
# Test Class: Connection Verification
# ============================================================================


class TestConnectionVerification:
    """Tests for connection verification."""

    @patch("requests.Session.get")
    def test_verify_connection_success(self, mock_get, mock_env_token, mock_quantum_computers):
        """Test successful connection verification."""
        mock_response = Mock()
        mock_response.json.return_value = mock_quantum_computers
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        manager = IQMBackendManager()
        assert manager.verify_connection() is True

    @patch("requests.Session.get")
    def test_verify_connection_no_backends(self, mock_get, mock_env_token):
        """Test connection verification with no backends available."""
        mock_response = Mock()
        mock_response.json.return_value = {"quantum_computers": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        manager = IQMBackendManager()
        assert manager.verify_connection() is False

    @patch("requests.Session.get")
    def test_verify_connection_api_error(self, mock_get, mock_env_token):
        """Test connection verification with API error."""
        mock_get.side_effect = RequestException("Network error")

        manager = IQMBackendManager()
        assert manager.verify_connection() is False


# ============================================================================
# Test Class: IQMRestBackend
# ============================================================================


class TestIQMRestBackend:
    """Tests for IQMRestBackend circuit execution."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock REST backend."""
        session = Mock()
        backend = IQMRestBackend(
            qc_id="qc-test-123",
            name="test-backend",
            session=session,
            base_url="https://test.api/v1",
        )
        return backend

    def test_backend_init(self, mock_backend):
        """Test backend initialization."""
        assert mock_backend.qc_id == "qc-test-123"
        assert mock_backend.name == "test-backend"
        assert mock_backend.base_url == "https://test.api/v1"

    def test_run_circuit_submission(self, mock_backend, mock_job_response):
        """Test circuit submission via run()."""
        mock_response = Mock()
        mock_response.json.return_value = mock_job_response
        mock_response.raise_for_status = Mock()
        mock_backend.session.post.return_value = mock_response

        # Create a mock circuit that will pass isinstance check
        try:
            from qiskit import QuantumCircuit
            mock_circuit = QuantumCircuit(2)
        except ImportError:
            # If qiskit not available, create a mock
            mock_circuit = Mock()
            mock_circuit.name = "test_circuit"
            mock_circuit.data = []
            mock_circuit.qubits = []

        job = mock_backend.run(mock_circuit, shots=1024)

        assert isinstance(job, IQMRestJob)
        assert job._job_id == "job-abc123"

    def test_backend_configuration(self, mock_backend):
        """Test backend configuration."""
        config = mock_backend.configuration()

        assert config.backend_name == "test-backend"
        assert config.n_qubits == 20
        assert "r" in config.basis_gates
        assert "cz" in config.basis_gates

    def test_gate_to_r_params_conversion(self, mock_backend):
        """Test conversion of standard gates to R gate parameters."""
        import math

        # Test X gate
        params_x = mock_backend._gate_to_r_params("x")
        assert params_x["theta"] == math.pi
        assert params_x["phi"] == 0.0

        # Test Y gate
        params_y = mock_backend._gate_to_r_params("y")
        assert params_y["theta"] == math.pi
        assert params_y["phi"] == math.pi / 2

        # Test S gate
        params_s = mock_backend._gate_to_r_params("s")
        assert params_s["theta"] == math.pi / 2

    def test_backend_repr(self, mock_backend):
        """Test backend string representation."""
        repr_str = repr(mock_backend)

        assert "IQMRestBackend" in repr_str
        assert "test-backend" in repr_str


# ============================================================================
# Test Class: IQMRestJob
# ============================================================================


class TestIQMRestJob:
    """Tests for IQMRestJob status tracking and result retrieval."""

    @pytest.fixture
    def mock_job(self):
        """Create mock job handle."""
        session = Mock()
        job = IQMRestJob(
            job_id="job-test-123",
            session=session,
            base_url="https://test.api/v1",
            backend_name="test-backend",
        )
        return job

    def test_job_init(self, mock_job):
        """Test job initialization."""
        assert mock_job._job_id == "job-test-123"
        assert mock_job.backend_name == "test-backend"
        assert mock_job._cached_status is None
        assert mock_job._cached_result is None

    def test_job_id_method(self, mock_job):
        """Test job_id() method."""
        assert mock_job.job_id() == "job-test-123"

    def test_status_query(self, mock_job, mock_job_status_done):
        """Test querying job status."""
        mock_response = Mock()
        mock_response.json.return_value = mock_job_status_done
        mock_response.raise_for_status = Mock()
        mock_job.session.get.return_value = mock_response

        status = mock_job.status()

        assert status == "done"
        assert mock_job._cached_status == "done"

    def test_status_query_api_error(self, mock_job):
        """Test status query with API error."""
        mock_job.session.get.side_effect = RequestException("Network error")

        with pytest.raises(RuntimeError, match="Status query failed"):
            mock_job.status()

    def test_done_method_completed(self, mock_job):
        """Test done() method for completed job."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "done"}
        mock_response.raise_for_status = Mock()
        mock_job.session.get.return_value = mock_response

        assert mock_job.done() is True

    def test_done_method_running(self, mock_job):
        """Test done() method for running job."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "running"}
        mock_response.raise_for_status = Mock()
        mock_job.session.get.return_value = mock_response

        assert mock_job.done() is False

    def test_result_with_timeout(self, mock_job, mock_job_status_done, mock_job_result):
        """Test result retrieval with timeout."""
        # Mock status and result calls
        status_response = Mock()
        status_response.json.return_value = mock_job_status_done
        status_response.raise_for_status = Mock()

        result_response = Mock()
        result_response.json.return_value = mock_job_result
        result_response.raise_for_status = Mock()

        mock_job.session.get.side_effect = [status_response, result_response]

        result = mock_job.result(timeout=10.0)

        assert isinstance(result, IQMResult)
        assert result.job_id == "job-test-123"
        assert result.success is True

    def test_result_cached(self, mock_job):
        """Test that result is cached after first retrieval."""
        # Set cached result
        cached_result = IQMResult(
            job_id="job-test-123", backend_name="test", data={}, success=True
        )
        mock_job._cached_result = cached_result

        result = mock_job.result()

        # Should return cached result without API call
        assert result == cached_result
        mock_job.session.get.assert_not_called()

    def test_cancel_job(self, mock_job):
        """Test job cancellation."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_job.session.post.return_value = mock_response

        success = mock_job.cancel()

        assert success is True
        assert mock_job._cached_status == IQMRestJob.STATUS_ABORTED
        mock_job.session.post.assert_called_once()

    def test_cancel_job_failure(self, mock_job):
        """Test job cancellation failure."""
        mock_job.session.post.side_effect = RequestException("Cancel failed")

        success = mock_job.cancel()

        assert success is False

    def test_job_repr(self, mock_job):
        """Test job string representation."""
        mock_job._cached_status = "running"
        repr_str = repr(mock_job)

        assert "IQMRestJob" in repr_str
        assert "running" in repr_str


# ============================================================================
# Test Class: IQMResult
# ============================================================================


class TestIQMResult:
    """Tests for IQMResult data parsing."""

    @pytest.fixture
    def sample_result(self, mock_job_result):
        """Create sample result object."""
        return IQMResult(
            job_id="job-test-123",
            backend_name="test-backend",
            data=mock_job_result,
            success=True,
        )

    def test_result_init(self, sample_result):
        """Test result initialization."""
        assert sample_result.job_id == "job-test-123"
        assert sample_result.backend_name == "test-backend"
        assert sample_result.success is True

    def test_get_counts(self, sample_result):
        """Test extracting measurement counts."""
        counts = sample_result.get_counts()

        # From mock data: [0,0], [0,1], [1,0], [1,1], [0,0], [0,1]
        assert counts["00"] == 2
        assert counts["01"] == 2
        assert counts["10"] == 1
        assert counts["11"] == 1

    def test_get_counts_empty_data(self):
        """Test get_counts with no measurement data."""
        result = IQMResult(
            job_id="job-test", backend_name="test", data={"measurements": []}, success=True
        )

        counts = result.get_counts()
        assert counts == {}

    def test_get_memory(self, sample_result):
        """Test extracting individual shot results."""
        memory = sample_result.get_memory()

        assert len(memory) == 6
        assert memory[0] == "00"
        assert memory[1] == "01"
        assert memory[2] == "10"

    def test_get_memory_empty_data(self):
        """Test get_memory with no measurement data."""
        result = IQMResult(
            job_id="job-test", backend_name="test", data={"measurements": []}, success=True
        )

        memory = result.get_memory()
        assert memory == []

    def test_result_repr(self, sample_result):
        """Test result string representation."""
        repr_str = repr(sample_result)

        assert "IQMResult" in repr_str
        assert "SUCCESS" in repr_str
        assert "test-backend" in repr_str


# ============================================================================
# Test Class: Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @patch("requests.Session.get")
    def test_fetch_quantum_computers_network_error(self, mock_get, mock_env_token):
        """Test handling of network errors during backend fetch."""
        mock_get.side_effect = RequestException("Connection timeout")

        manager = IQMBackendManager()

        with pytest.raises(RequestException):
            manager._fetch_quantum_computers()

    def test_job_submission_api_error(self, mock_env_token):
        """Test handling of API errors during job submission."""
        session = Mock()
        session.post.side_effect = RequestException("API Error")

        backend = IQMRestBackend(
            qc_id="test", name="test", session=session, base_url="https://test.api/v1"
        )

        # Create a mock circuit
        try:
            from qiskit import QuantumCircuit
            mock_circuit = QuantumCircuit(2)
        except ImportError:
            mock_circuit = Mock()
            mock_circuit.data = []

        with pytest.raises(RuntimeError, match="Failed to submit job"):
            backend.run(mock_circuit)

    @patch("requests.Session.get")
    @patch("time.sleep", return_value=None)  # Speed up test
    def test_result_timeout_exceeded(self, mock_sleep, mock_get, mock_env_token):
        """Test result retrieval timeout."""
        # Mock perpetually running job
        mock_response = Mock()
        mock_response.json.return_value = {"status": "running"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        session = Mock()
        session.get.return_value = mock_response

        job = IQMRestJob(
            job_id="job-test", session=session, base_url="https://test.api/v1", backend_name="test"
        )

        with pytest.raises(RuntimeError, match="Job timeout"):
            job.result(timeout=0.1)


# ============================================================================
# Summary Statistics
# ============================================================================

# Test count: 28 tests
# Coverage areas:
# - IQMBackendManager initialization (4 tests)
# - Backend selection and topology (7 tests)
# - Connection verification (3 tests)
# - IQMRestBackend operations (5 tests)
# - IQMRestJob status and results (7 tests)
# - IQMResult parsing (6 tests)
# - Error handling (3 tests)
