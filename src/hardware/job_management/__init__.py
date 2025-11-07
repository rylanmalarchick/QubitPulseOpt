"""
Job Management Module
=====================

This module provides asynchronous job submission and management for IQM hardware,
enabling efficient batch execution and resilient recovery from failures.

Classes
-------
AsyncJobManager : Main job queue and retrieval manager
JobRecord : Individual job tracking record
JobStatus : Enum for job states

Example
-------
>>> from src.hardware.job_management import AsyncJobManager, JobStatus
>>> from src.hardware import IQMBackendManager
>>>
>>> backend_mgr = IQMBackendManager()
>>> backend = backend_mgr.get_backend()
>>> manager = AsyncJobManager(backend, session_id="phase5_validation")
>>>
>>> # Submit jobs
>>> job_id = manager.submit_job(circuit, shots=1024, metadata={"phase": 1})
>>>
>>> # Monitor progress
>>> while not manager.all_complete([job_id]):
...     summary = manager.get_status_summary([job_id])
...     print(f"Progress: {summary['completed']}/{summary['total']}")
...     time.sleep(30)
>>>
>>> # Retrieve results
>>> result = manager.get_result(job_id)
>>>
>>> # Save session for resilience
>>> manager.save_session("session.json")
>>>
>>> # Restore later
>>> manager = AsyncJobManager.load_session("session.json", backend)
"""

from .async_job_manager import AsyncJobManager, JobRecord, JobStatus

__all__ = ["AsyncJobManager", "JobRecord", "JobStatus"]
