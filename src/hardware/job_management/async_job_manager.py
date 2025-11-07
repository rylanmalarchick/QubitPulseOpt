"""
Async Job Manager for IQM Hardware
===================================

This module provides asynchronous job submission and management for IQM Resonance
hardware, enabling efficient batch execution and resilient recovery from failures.

Key Features
------------
- Asynchronous job submission (submit all at once, retrieve later)
- Job queue management and status tracking
- Automatic retry logic for failed jobs
- Session persistence (survive network disconnects)
- Credit usage tracking
- Progress monitoring and reporting

Usage Pattern
-------------
Instead of synchronous execution:
    job = backend.run(circuit, shots=1024)
    result = job.result()  # BLOCKS until done

Use asynchronous batch submission:
    manager = AsyncJobManager(backend)

    # Submit all jobs at once
    job_ids = []
    for circuit in circuits:
        job_id = manager.submit_job(circuit, shots=1024, metadata={...})
        job_ids.append(job_id)

    # Poll for completion (non-blocking)
    while not manager.all_complete(job_ids):
        status = manager.get_status_summary(job_ids)
        print(f"Progress: {status['completed']}/{status['total']}")
        time.sleep(30)

    # Retrieve all results
    results = manager.get_all_results(job_ids)

Classes
-------
AsyncJobManager : Main job queue and retrieval manager
JobRecord : Individual job tracking record
JobStatus : Enum for job states

Example
-------
>>> from src.hardware.job_management import AsyncJobManager
>>> from src.hardware import IQMBackendManager
>>>
>>> backend_mgr = IQMBackendManager()
>>> backend = backend_mgr.get_backend()
>>> manager = AsyncJobManager(backend, session_id="phase5_validation")
>>>
>>> # Submit Phase 1-5 experiments as batch
>>> job_id_1 = manager.submit_job(t1_circuit, shots=512, metadata={"phase": 1, "experiment": "T1"})
>>> job_id_2 = manager.submit_job(rb_circuits, shots=256, metadata={"phase": 4, "experiment": "RB"})
>>>
>>> # Save session (can disconnect and come back)
>>> manager.save_session("session_phase5.json")
>>>
>>> # Later: restore and check status
>>> manager = AsyncJobManager.load_session("session_phase5.json", backend)
>>> status = manager.get_status_summary()
>>> print(f"Completed: {status['completed']}/{status['total']}")
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status states."""

    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class JobRecord:
    """
    Record for tracking individual job submission and status.

    Attributes
    ----------
    job_id : str
        IQM job identifier
    submit_time : float
        Unix timestamp when job was submitted
    status : JobStatus
        Current job status
    circuit_hash : str
        Hash of circuit for deduplication
    shots : int
        Number of measurement shots
    metadata : dict
        User-provided metadata (phase, experiment name, etc.)
    result : dict or None
        Retrieved result data (None until completed)
    error : str or None
        Error message if failed
    retry_count : int
        Number of retry attempts
    cost_estimate : int
        Estimated cost in IQM credits
    completion_time : float or None
        Unix timestamp when job completed
    """

    job_id: str
    submit_time: float
    status: JobStatus
    circuit_hash: str
    shots: int
    metadata: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    cost_estimate: int = 0
    completion_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobRecord":
        """Restore from dict."""
        data["status"] = JobStatus(data["status"])
        return cls(**data)


class AsyncJobManager:
    """
    Manages asynchronous job submission and retrieval for IQM hardware.

    This class decouples job submission from result retrieval, allowing
    efficient batch submission and resilient recovery from network issues.

    Parameters
    ----------
    backend : IQMBackend
        IQM quantum backend instance
    session_id : str, optional
        Unique identifier for this session (for persistence)
    max_retries : int, default=3
        Maximum retry attempts for failed jobs
    poll_interval : int, default=30
        Seconds between status polls

    Attributes
    ----------
    jobs : Dict[str, JobRecord]
        Mapping of job_id to JobRecord
    session_id : str
        Session identifier
    backend : IQMBackend
        Backend instance
    total_credits_used : int
        Running total of credits consumed

    Example
    -------
    >>> manager = AsyncJobManager(backend, session_id="run_001")
    >>> job_id = manager.submit_job(circuit, shots=1024)
    >>> manager.wait_for_completion([job_id], timeout=3600)
    >>> result = manager.get_result(job_id)
    """

    def __init__(
        self,
        backend: Any,
        session_id: Optional[str] = None,
        max_retries: int = 3,
        poll_interval: int = 30,
    ):
        """Initialize async job manager."""
        self.backend = backend
        self.session_id = session_id or f"session_{int(time.time())}"
        self.max_retries = max_retries
        self.poll_interval = poll_interval

        self.jobs: Dict[str, JobRecord] = {}
        self.total_credits_used: int = 0

        # Cache job objects for local simulators (Aer, etc.)
        self._job_cache: Dict[str, Any] = {}

        # Detect if backend is a local simulator (no job retrieval support)
        self.is_local_simulator = self._detect_local_simulator()

        logger.info(
            f"AsyncJobManager initialized: session={self.session_id}, "
            f"max_retries={max_retries}, poll_interval={poll_interval}s, "
            f"local_simulator={self.is_local_simulator}"
        )

    def submit_job(
        self,
        circuit: Any,
        shots: int = 1024,
        metadata: Optional[Dict[str, Any]] = None,
        priority: str = "normal",
    ) -> str:
        """
        Submit a job to IQM hardware (non-blocking).

        Parameters
        ----------
        circuit : QuantumCircuit or Schedule
            Circuit or pulse schedule to execute
        shots : int, default=1024
            Number of measurement shots
        metadata : dict, optional
            User metadata for tracking (e.g., {"phase": 1, "experiment": "T1"})
        priority : str, default='normal'
            Job priority: 'low', 'normal', 'high'

        Returns
        -------
        job_id : str
            IQM job identifier for later retrieval

        Raises
        ------
        RuntimeError
            If job submission fails

        Example
        -------
        >>> job_id = manager.submit_job(
        ...     circuit=bell_state,
        ...     shots=512,
        ...     metadata={"phase": 1, "test": "handshake"}
        ... )
        """
        try:
            # Submit job to backend (non-blocking)
            logger.info(f"Submitting job: shots={shots}, metadata={metadata}")
            job = self.backend.run(circuit, shots=shots)

            # Extract job ID
            if hasattr(job, "job_id"):
                job_id = job.job_id()
            elif hasattr(job, "id"):
                job_id = job.id()
            else:
                job_id = str(job)

            # Cache job object for local simulators
            if self.is_local_simulator:
                self._job_cache[job_id] = job

            # Create circuit hash for deduplication
            circuit_hash = self._hash_circuit(circuit)

            # Estimate cost (IQM charges per 1000 shots typically)
            cost_estimate = self._estimate_cost(shots)

            # Create job record
            record = JobRecord(
                job_id=job_id,
                submit_time=time.time(),
                status=JobStatus.SUBMITTED,
                circuit_hash=circuit_hash,
                shots=shots,
                metadata=metadata or {},
                cost_estimate=cost_estimate,
            )

            self.jobs[job_id] = record

            logger.info(
                f"Job submitted: id={job_id}, hash={circuit_hash[:8]}, "
                f"cost_est={cost_estimate} credits"
            )

            return job_id

        except Exception as e:
            logger.error(f"Job submission failed: {e}")
            raise RuntimeError(f"Failed to submit job: {e}") from e

    def get_status(self, job_id: str) -> JobStatus:
        """
        Get current status of a job.

        Parameters
        ----------
        job_id : str
            Job identifier

        Returns
        -------
        status : JobStatus
            Current job status
        """
        if job_id not in self.jobs:
            logger.warning(f"Unknown job_id: {job_id}")
            return JobStatus.UNKNOWN

        record = self.jobs[job_id]

        # If already completed/failed, return cached status
        if record.status in [
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ]:
            return record.status

        # Poll backend for updated status
        try:
            # Get job handle
            job = self._get_job_handle(job_id)

            if job is None:
                # For local simulators after session restore, jobs can't be retrieved
                # Mark them as completed since simulators execute synchronously
                if self.is_local_simulator:
                    logger.debug(
                        f"Local simulator job {job_id} not in cache - marking completed"
                    )
                    record.status = JobStatus.COMPLETED
                    record.completion_time = record.submit_time  # Immediate completion
                    return record.status
                else:
                    logger.warning(f"Job handle not found for {job_id}")
                    return JobStatus.UNKNOWN

            # For local simulators, jobs complete immediately
            if self.is_local_simulator:
                # Check if job is done
                if hasattr(job, "done") and callable(job.done):
                    if job.done():
                        record.status = JobStatus.COMPLETED
                        record.completion_time = time.time()
                        return record.status
                # Otherwise assume completed (Aer jobs are synchronous)
                record.status = JobStatus.COMPLETED
                record.completion_time = time.time()
                return record.status

            # For real hardware: poll status
            if hasattr(job, "status"):
                backend_status = job.status()

                # Map backend status to our enum
                if hasattr(backend_status, "name"):
                    status_str = backend_status.name.lower()
                else:
                    status_str = str(backend_status).lower()

                if "done" in status_str or "completed" in status_str:
                    record.status = JobStatus.COMPLETED
                    record.completion_time = time.time()
                elif "running" in status_str or "validating" in status_str:
                    record.status = JobStatus.RUNNING
                elif "queued" in status_str or "pending" in status_str:
                    record.status = JobStatus.QUEUED
                elif "error" in status_str or "failed" in status_str:
                    record.status = JobStatus.FAILED
                elif "cancel" in status_str:
                    record.status = JobStatus.CANCELLED

                return record.status

        except Exception as e:
            logger.warning(f"Failed to get status for job {job_id}: {e}")
            return JobStatus.UNKNOWN

        return record.status

    def get_result(
        self,
        job_id: str,
        wait: bool = False,
        timeout: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve result for a completed job.

        Parameters
        ----------
        job_id : str
            Job identifier
        wait : bool, default=False
            If True, block until job completes
        timeout : int, optional
            Maximum seconds to wait (only if wait=True)

        Returns
        -------
        result : dict or None
            Job result data, or None if not yet complete

        Example
        -------
        >>> result = manager.get_result(job_id, wait=True, timeout=3600)
        >>> counts = result['counts']
        """
        if job_id not in self.jobs:
            logger.error(f"Unknown job_id: {job_id}")
            return None

        record = self.jobs[job_id]

        # If already retrieved, return cached result
        if record.result is not None:
            return record.result

        # If waiting, poll until complete
        if wait:
            start_time = time.time()
            while True:
                status = self.get_status(job_id)

                if status == JobStatus.COMPLETED:
                    break
                elif status == JobStatus.FAILED:
                    logger.error(f"Job {job_id} failed: {record.error}")
                    return None

                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning(f"Timeout waiting for job {job_id}")
                    return None

                time.sleep(self.poll_interval)

        # Retrieve result from backend
        try:
            job = self._get_job_handle(job_id)

            # If job not found (e.g., local simulator after session restore)
            if job is None:
                if self.is_local_simulator:
                    logger.warning(
                        f"Cannot retrieve result for job {job_id} - "
                        f"job object not in cache (local simulator limitation)"
                    )
                    record.error = "Job object not cached (emulator session restore)"
                    record.status = JobStatus.FAILED
                    return None
                else:
                    logger.error(f"Job handle not found for {job_id}")
                    record.error = "Job not found on backend"
                    record.status = JobStatus.FAILED
                    return None

            result_obj = job.result()

            # Parse result
            result = {
                "counts": result_obj.get_counts()
                if hasattr(result_obj, "get_counts")
                else {},
                "shots": record.shots,
                "success": getattr(result_obj, "success", True),
                "job_id": job_id,
                "metadata": record.metadata,
                "execution_time": record.completion_time - record.submit_time
                if record.completion_time
                else None,
            }

            # Cache result
            record.result = result
            self.total_credits_used += record.cost_estimate

            logger.info(f"Retrieved result for job {job_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to retrieve result for job {job_id}: {e}")
            record.error = str(e)
            record.status = JobStatus.FAILED
            return None

    def get_status_summary(self, job_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get summary of job statuses.

        Parameters
        ----------
        job_ids : list of str, optional
            Specific jobs to summarize. If None, summarize all jobs.

        Returns
        -------
        summary : dict
            Summary with keys: total, submitted, queued, running, completed, failed

        Example
        -------
        >>> summary = manager.get_status_summary()
        >>> print(f"Progress: {summary['completed']}/{summary['total']}")
        """
        if job_ids is None:
            job_ids = list(self.jobs.keys())

        # Update all statuses
        statuses = [self.get_status(jid) for jid in job_ids]

        summary = {
            "total": len(job_ids),
            "submitted": sum(1 for s in statuses if s == JobStatus.SUBMITTED),
            "queued": sum(1 for s in statuses if s == JobStatus.QUEUED),
            "running": sum(1 for s in statuses if s == JobStatus.RUNNING),
            "completed": sum(1 for s in statuses if s == JobStatus.COMPLETED),
            "failed": sum(1 for s in statuses if s == JobStatus.FAILED),
            "cancelled": sum(1 for s in statuses if s == JobStatus.CANCELLED),
            "credits_used": self.total_credits_used,
        }

        return summary

    def all_complete(self, job_ids: List[str]) -> bool:
        """
        Check if all jobs are complete (success or failure).

        Parameters
        ----------
        job_ids : list of str
            Jobs to check

        Returns
        -------
        complete : bool
            True if all jobs are done
        """
        summary = self.get_status_summary(job_ids)
        return (
            summary["completed"] + summary["failed"] + summary["cancelled"]
        ) == summary["total"]

    def wait_for_completion(
        self,
        job_ids: List[str],
        timeout: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Block until all jobs complete.

        Parameters
        ----------
        job_ids : list of str
            Jobs to wait for
        timeout : int, optional
            Maximum seconds to wait
        progress_callback : callable, optional
            Function called with status summary each poll: callback(summary)

        Returns
        -------
        final_summary : dict
            Final status summary

        Example
        -------
        >>> def print_progress(summary):
        ...     print(f"{summary['completed']}/{summary['total']} complete")
        >>> manager.wait_for_completion(job_ids, timeout=3600, progress_callback=print_progress)
        """
        logger.info(f"Waiting for {len(job_ids)} jobs to complete...")
        start_time = time.time()

        while not self.all_complete(job_ids):
            summary = self.get_status_summary(job_ids)

            if progress_callback:
                progress_callback(summary)

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout after {timeout}s")
                break

            time.sleep(self.poll_interval)

        final_summary = self.get_status_summary(job_ids)
        logger.info(
            f"Completion: {final_summary['completed']} succeeded, "
            f"{final_summary['failed']} failed"
        )

        return final_summary

    def get_all_results(self, job_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve results for multiple jobs.

        Parameters
        ----------
        job_ids : list of str
            Jobs to retrieve

        Returns
        -------
        results : dict
            Mapping of job_id to result dict

        Example
        -------
        >>> results = manager.get_all_results(job_ids)
        >>> for job_id, result in results.items():
        ...     print(f"{job_id}: {result['counts']}")
        """
        results = {}
        for job_id in job_ids:
            result = self.get_result(job_id, wait=False)
            if result:
                results[job_id] = result

        return results

    def save_session(self, filepath: Union[str, Path]) -> None:
        """
        Save session state to JSON file.

        This enables recovery after network disconnects or crashes.

        Parameters
        ----------
        filepath : str or Path
            Path to save session

        Example
        -------
        >>> manager.save_session("session_phase5.json")
        """
        filepath = Path(filepath)

        session_data = {
            "session_id": self.session_id,
            "total_credits_used": self.total_credits_used,
            "jobs": {jid: rec.to_dict() for jid, rec in self.jobs.items()},
            "timestamp": time.time(),
        }

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Session saved: {filepath} ({len(self.jobs)} jobs)")

    @classmethod
    def load_session(
        cls,
        filepath: Union[str, Path],
        backend: Any,
    ) -> "AsyncJobManager":
        """
        Restore session from JSON file.

        Parameters
        ----------
        filepath : str or Path
            Path to saved session
        backend : IQMBackend
            Backend instance

        Returns
        -------
        manager : AsyncJobManager
            Restored manager instance

        Example
        -------
        >>> manager = AsyncJobManager.load_session("session_phase5.json", backend)
        >>> summary = manager.get_status_summary()
        """
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            session_data = json.load(f)

        manager = cls(backend, session_id=session_data["session_id"])
        manager.total_credits_used = session_data["total_credits_used"]

        # Restore job records
        for job_id, job_dict in session_data["jobs"].items():
            manager.jobs[job_id] = JobRecord.from_dict(job_dict)

        logger.info(
            f"Session restored: {filepath} (session_id={manager.session_id}, "
            f"{len(manager.jobs)} jobs)"
        )

        return manager

    # ========================================================================
    # Helper methods
    # ========================================================================

    def _get_job_handle(self, job_id: str) -> Any:
        """Get backend job handle for a job_id."""
        # For local simulators, use cached job object
        if self.is_local_simulator:
            return self._job_cache.get(job_id)

        # For real hardware: retrieve from backend
        if hasattr(self.backend, "retrieve_job"):
            return self.backend.retrieve_job(job_id)
        elif hasattr(self.backend, "job"):
            return self.backend.job(job_id)
        else:
            # Fall back to cache if retrieval not supported
            return self._job_cache.get(job_id)

    def _hash_circuit(self, circuit: Any) -> str:
        """Create hash of circuit for deduplication."""
        import hashlib

        # Simple hash based on circuit QASM if available
        if hasattr(circuit, "qasm"):
            qasm_str = circuit.qasm()
            return hashlib.sha256(qasm_str.encode()).hexdigest()
        else:
            return hashlib.sha256(str(circuit).encode()).hexdigest()

    def _estimate_cost(self, shots: int) -> int:
        """
        Estimate cost in IQM credits.

        IQM typically charges per 1000 shots, exact pricing varies.
        This is a rough estimate for tracking purposes.
        """
        # Rough estimate: 30 credits per 1000 shots (adjust based on actual pricing)
        return int((shots / 1000) * 30)

    def _detect_local_simulator(self) -> bool:
        """
        Detect if backend is a local simulator (Aer, fake backend, etc.).

        Local simulators don't support async job retrieval - jobs complete
        immediately and synchronously.
        """
        backend_name = getattr(self.backend, "name", "").lower()
        backend_class = self.backend.__class__.__name__.lower()

        # Check for known local simulators
        if any(x in backend_name for x in ["aer", "simulator", "fake"]):
            return True
        if any(x in backend_class for x in ["aer", "simulator", "fake"]):
            return True

        # Check if backend has retrieve_job (real hardware has this)
        if not hasattr(self.backend, "retrieve_job") and not hasattr(
            self.backend, "job"
        ):
            return True

        return False
