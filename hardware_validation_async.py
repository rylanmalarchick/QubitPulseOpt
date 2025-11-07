#!/usr/bin/env python3
"""
Hardware Validation (Async Job Submission)
===========================================

This is an improved version of hardware_validation.py that uses asynchronous
job submission for optimal resource utilization during IQM Resonance sessions.

KEY IMPROVEMENTS vs Synchronous Version
---------------------------------------
1. **Batch Submission**: Submit all experiments at once, retrieve results later
2. **Network Resilience**: Can disconnect and reconnect without losing jobs
3. **Efficient Resource Use**: Don't block on job.result() - hardware keeps running
4. **Session Persistence**: Save/restore session state to survive crashes
5. **Better Monitoring**: Real-time progress tracking across all phases

USAGE
-----
# Standard run (submit all jobs, wait for completion)
python hardware_validation_async.py

# Submit jobs only (can disconnect after submission)
python hardware_validation_async.py --submit-only

# Resume and collect results (after submission)
python hardware_validation_async.py --resume session_phase5.json

# Dry-run (emulator test)
python hardware_validation_async.py --dry-run

WORKFLOW
--------
Phase 1: Submit characterization experiments (T1, T2)
Phase 2: Submit DRAG vs Gaussian pulse comparisons
Phase 3: Submit custom pulse executions
Phase 4: Submit benchmarking experiments (RB, IRB)
Phase 5: Collect all results and generate report

All jobs are submitted in batches, then polled asynchronously for completion.

Example Session
---------------
# Start 6-hour hardware session
$ python hardware_validation_async.py --submit-only
> Submitted 47 jobs across 5 phases
> Session saved: session_20240115_1430.json
> You can disconnect now - jobs are running on hardware

# 2 hours later: check progress
$ python hardware_validation_async.py --resume session_20240115_1430.json --status-only
> Progress: 32/47 jobs complete (68%)
> Estimated remaining time: 45 minutes
> Credits used: 1240 / ~1800 estimated

# Final retrieval
$ python hardware_validation_async.py --resume session_20240115_1430.json
> All jobs complete! Generating report...
> Report: hardware_validation_results/HARDWARE_FIDELITY_REPORT.md
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.hardware.iqm_backend import IQMBackendManager
from src.hardware.job_management.async_job_manager import AsyncJobManager, JobStatus
from src.logging_utils import setup_logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AsyncHardwareValidator:
    """
    Hardware validation orchestrator using async job submission.

    This class manages the full Phase 1-5 validation workflow using
    asynchronous job submission for optimal hardware utilization.

    Parameters
    ----------
    use_emulator : bool, default=False
        Use emulator backend instead of real hardware
    session_id : str, optional
        Session identifier for persistence
    output_dir : Path, optional
        Directory for results and reports

    Attributes
    ----------
    backend_manager : IQMBackendManager
        Hardware backend manager
    job_manager : AsyncJobManager
        Async job queue manager
    phase_jobs : dict
        Mapping of phase number to list of job IDs
    results_dir : Path
        Output directory for results
    """

    def __init__(
        self,
        use_emulator: bool = False,
        session_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        """Initialize async hardware validator."""
        self.use_emulator = use_emulator
        self.session_id = (
            session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Setup output directory
        self.results_dir = output_dir or Path("hardware_validation_results")
        self.results_dir.mkdir(exist_ok=True)

        # Initialize backend
        logger.info("Initializing IQM backend...")
        self.backend_manager = IQMBackendManager()
        self.backend = self.backend_manager.get_backend(use_emulator=use_emulator)

        # Initialize async job manager
        logger.info("Initializing async job manager...")
        self.job_manager = AsyncJobManager(
            backend=self.backend,
            session_id=self.session_id,
            max_retries=3,
            poll_interval=30,
        )

        # Track jobs by phase
        self.phase_jobs: Dict[int, List[str]] = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
        }

        logger.info(f"AsyncHardwareValidator initialized: session={self.session_id}")

    def submit_all_phases(self, target_qubit: str = "QB1") -> Dict[str, Any]:
        """
        Submit all Phase 1-5 experiments as batch jobs.

        This is the key optimization: submit everything at once, then poll.

        Parameters
        ----------
        target_qubit : str, default='QB1'
            Target qubit for validation

        Returns
        -------
        summary : dict
            Summary of submitted jobs by phase
        """
        logger.info("=" * 80)
        logger.info("BATCH JOB SUBMISSION: Phases 1-5")
        logger.info("=" * 80)

        # Phase 1: Hardware handshake and characterization
        logger.info("\nPhase 1: Submitting characterization experiments...")
        phase1_jobs = self._submit_phase1(target_qubit)
        self.phase_jobs[1] = phase1_jobs
        logger.info(f"  ✓ Submitted {len(phase1_jobs)} Phase 1 jobs")

        # Phase 2: DRAG vs Gaussian optimization
        logger.info("\nPhase 2: Submitting pulse optimization experiments...")
        phase2_jobs = self._submit_phase2(target_qubit)
        self.phase_jobs[2] = phase2_jobs
        logger.info(f"  ✓ Submitted {len(phase2_jobs)} Phase 2 jobs")

        # Phase 3: Custom pulse execution
        logger.info("\nPhase 3: Submitting custom pulse experiments...")
        phase3_jobs = self._submit_phase3(target_qubit)
        self.phase_jobs[3] = phase3_jobs
        logger.info(f"  ✓ Submitted {len(phase3_jobs)} Phase 3 jobs")

        # Phase 4: Benchmarking (RB, IRB)
        logger.info("\nPhase 4: Submitting benchmarking experiments...")
        phase4_jobs = self._submit_phase4(target_qubit)
        self.phase_jobs[4] = phase4_jobs
        logger.info(f"  ✓ Submitted {len(phase4_jobs)} Phase 4 jobs")

        # Phase 5: DRAG vs Gaussian fidelity comparison
        logger.info("\nPhase 5: Submitting final validation experiments...")
        phase5_jobs = self._submit_phase5(target_qubit)
        self.phase_jobs[5] = phase5_jobs
        logger.info(f"  ✓ Submitted {len(phase5_jobs)} Phase 5 jobs")

        # Summary
        total_jobs = sum(len(jobs) for jobs in self.phase_jobs.values())
        summary = {
            "session_id": self.session_id,
            "total_jobs": total_jobs,
            "phase_breakdown": {
                phase: len(jobs) for phase, jobs in self.phase_jobs.items()
            },
            "target_qubit": target_qubit,
            "submission_time": datetime.now().isoformat(),
        }

        logger.info("\n" + "=" * 80)
        logger.info(f"BATCH SUBMISSION COMPLETE: {total_jobs} jobs submitted")
        logger.info("=" * 80)

        return summary

    def wait_and_collect_results(
        self,
        timeout: Optional[int] = None,
        save_interval: int = 300,
    ) -> Dict[str, Any]:
        """
        Wait for all jobs to complete and collect results.

        Parameters
        ----------
        timeout : int, optional
            Maximum seconds to wait (default: no timeout)
        save_interval : int, default=300
            Save session state every N seconds

        Returns
        -------
        results : dict
            All results organized by phase
        """
        logger.info("\nWaiting for job completion...")

        all_job_ids = []
        for jobs in self.phase_jobs.values():
            all_job_ids.extend(jobs)

        last_save = time.time()

        def progress_callback(summary):
            """Print progress and save session periodically."""
            nonlocal last_save

            logger.info(
                f"Progress: {summary['completed']}/{summary['total']} complete "
                f"({summary['running']} running, {summary['queued']} queued, "
                f"{summary['failed']} failed)"
            )
            logger.info(f"Credits used: {summary['credits_used']}")

            # Periodic save
            if time.time() - last_save > save_interval:
                self.save_session()
                last_save = time.time()

        # Wait for completion
        final_summary = self.job_manager.wait_for_completion(
            job_ids=all_job_ids,
            timeout=timeout,
            progress_callback=progress_callback,
        )

        logger.info(f"\nAll jobs complete!")
        logger.info(f"  Total: {final_summary['total']}")
        logger.info(f"  Completed: {final_summary['completed']}")
        logger.info(f"  Failed: {final_summary['failed']}")
        logger.info(f"  Total credits: {final_summary['credits_used']}")

        # Collect results
        logger.info("\nCollecting results...")
        results = {}
        for phase, job_ids in self.phase_jobs.items():
            phase_results = self.job_manager.get_all_results(job_ids)
            results[f"phase_{phase}"] = phase_results
            logger.info(f"  Phase {phase}: {len(phase_results)} results retrieved")

        return results

    def generate_report(self, results: Dict[str, Any]) -> Path:
        """
        Generate hardware validation report.

        Parameters
        ----------
        results : dict
            Results from all phases

        Returns
        -------
        report_path : Path
            Path to generated report
        """
        report_path = self.results_dir / "HARDWARE_FIDELITY_REPORT.md"

        with open(report_path, "w") as f:
            f.write("# IQM Hardware Validation Report\n\n")
            f.write(f"**Session ID:** {self.session_id}\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n")
            f.write(f"**Emulator Mode:** {self.use_emulator}\n\n")

            f.write("---\n\n")

            # Phase summaries
            for phase in range(1, 6):
                phase_key = f"phase_{phase}"
                if phase_key in results:
                    phase_results = results[phase_key]

                    f.write(f"## Phase {phase} Results\n\n")
                    f.write(f"**Jobs Completed:** {len(phase_results)}\n\n")

                    # Detailed results
                    for job_id, result in phase_results.items():
                        metadata = result.get("metadata", {})
                        f.write(f"### Job: {job_id}\n")
                        f.write(
                            f"- **Experiment:** {metadata.get('experiment', 'N/A')}\n"
                        )
                        f.write(f"- **Shots:** {result.get('shots', 0)}\n")
                        f.write(f"- **Success:** {result.get('success', False)}\n")

                        # Counts summary
                        counts = result.get("counts", {})
                        if counts:
                            f.write(f"- **Measurement Counts:** {counts}\n")

                        f.write("\n")

                    f.write("\n")

            # Final summary
            f.write("---\n\n")
            f.write("## Summary\n\n")

            total_jobs = sum(len(results.get(f"phase_{i}", {})) for i in range(1, 6))
            f.write(f"- **Total Jobs:** {total_jobs}\n")
            f.write(f"- **Session Duration:** {self._get_session_duration()}\n")
            f.write(
                f"- **Total Credits Used:** {self.job_manager.total_credits_used}\n"
            )

        logger.info(f"Report generated: {report_path}")
        return report_path

    def save_session(self, filepath: Optional[Path] = None) -> Path:
        """
        Save session state to file.

        Parameters
        ----------
        filepath : Path, optional
            Path to save session. If None, auto-generate name.

        Returns
        -------
        filepath : Path
            Path where session was saved
        """
        if filepath is None:
            filepath = self.results_dir / f"{self.session_id}.json"

        # Save job manager state
        self.job_manager.save_session(filepath)

        # Add phase job mapping
        with open(filepath, "r") as f:
            session_data = json.load(f)

        session_data["phase_jobs"] = self.phase_jobs
        session_data["use_emulator"] = self.use_emulator

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Session saved: {filepath}")
        return filepath

    @classmethod
    def load_session(cls, filepath: Path) -> "AsyncHardwareValidator":
        """
        Restore session from file.

        Parameters
        ----------
        filepath : Path
            Path to saved session

        Returns
        -------
        validator : AsyncHardwareValidator
            Restored validator instance
        """
        with open(filepath, "r") as f:
            session_data = json.load(f)

        # Create new instance
        validator = cls(
            use_emulator=session_data.get("use_emulator", False),
            session_id=session_data["session_id"],
        )

        # Restore job manager
        validator.job_manager = AsyncJobManager.load_session(
            filepath, validator.backend
        )

        # Restore phase jobs
        validator.phase_jobs = {
            int(k): v for k, v in session_data.get("phase_jobs", {}).items()
        }

        logger.info(f"Session restored: {filepath}")
        return validator

    def get_status_summary(self) -> Dict[str, Any]:
        """Get current status summary across all phases."""
        all_jobs = []
        for jobs in self.phase_jobs.values():
            all_jobs.extend(jobs)

        overall_summary = self.job_manager.get_status_summary(all_jobs)

        # Per-phase summaries
        phase_summaries = {}
        for phase, jobs in self.phase_jobs.items():
            if jobs:
                phase_summaries[f"phase_{phase}"] = self.job_manager.get_status_summary(
                    jobs
                )

        return {
            "overall": overall_summary,
            "by_phase": phase_summaries,
            "session_id": self.session_id,
        }

    # ========================================================================
    # Phase-specific job submission methods
    # ========================================================================

    def _submit_phase1(self, target_qubit: str) -> List[str]:
        """Submit Phase 1: Hardware handshake & characterization."""
        from qiskit import QuantumCircuit

        job_ids = []

        # Job 1: Bell state (hardware handshake)
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        job_id = self.job_manager.submit_job(
            circuit=qc,
            shots=1024,
            metadata={"phase": 1, "experiment": "bell_state", "test": "handshake"},
        )
        job_ids.append(job_id)

        # Job 2: T1 characterization (placeholder - would use qiskit-experiments)
        # In production, this would submit a T1 experiment
        # For now, submit a simple identity circuit as placeholder
        qc_t1 = QuantumCircuit(1, 1)
        qc_t1.id(0)
        qc_t1.measure(0, 0)

        job_id = self.job_manager.submit_job(
            circuit=qc_t1,
            shots=512,
            metadata={
                "phase": 1,
                "experiment": "t1_characterization",
                "qubit": target_qubit,
            },
        )
        job_ids.append(job_id)

        return job_ids

    def _submit_phase2(self, target_qubit: str) -> List[str]:
        """Submit Phase 2: DRAG vs Gaussian optimization."""
        from qiskit import QuantumCircuit

        job_ids = []

        # Placeholder: Submit circuits for DRAG and Gaussian pulse comparison
        # In production, these would be actual optimized pulse schedules

        # DRAG pulse test
        qc_drag = QuantumCircuit(1, 1)
        qc_drag.x(0)  # Placeholder for DRAG pulse
        qc_drag.measure(0, 0)

        job_id = self.job_manager.submit_job(
            circuit=qc_drag,
            shots=1024,
            metadata={"phase": 2, "experiment": "drag_pulse", "qubit": target_qubit},
        )
        job_ids.append(job_id)

        # Gaussian pulse test
        qc_gauss = QuantumCircuit(1, 1)
        qc_gauss.x(0)  # Placeholder for Gaussian pulse
        qc_gauss.measure(0, 0)

        job_id = self.job_manager.submit_job(
            circuit=qc_gauss,
            shots=1024,
            metadata={
                "phase": 2,
                "experiment": "gaussian_pulse",
                "qubit": target_qubit,
            },
        )
        job_ids.append(job_id)

        return job_ids

    def _submit_phase3(self, target_qubit: str) -> List[str]:
        """Submit Phase 3: Custom pulse execution."""
        from qiskit import QuantumCircuit

        job_ids = []

        # Placeholder: Custom pulse execution
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)

        job_id = self.job_manager.submit_job(
            circuit=qc,
            shots=1024,
            metadata={"phase": 3, "experiment": "custom_pulse", "qubit": target_qubit},
        )
        job_ids.append(job_id)

        return job_ids

    def _submit_phase4(self, target_qubit: str) -> List[str]:
        """Submit Phase 4: Benchmarking experiments."""
        from qiskit import QuantumCircuit

        job_ids = []

        # Placeholder: RB experiment
        # In production, would use qiskit-experiments StandardRB
        qc_rb = QuantumCircuit(1, 1)
        qc_rb.x(0)
        qc_rb.measure(0, 0)

        job_id = self.job_manager.submit_job(
            circuit=qc_rb,
            shots=512,
            metadata={
                "phase": 4,
                "experiment": "randomized_benchmarking",
                "qubit": target_qubit,
            },
        )
        job_ids.append(job_id)

        return job_ids

    def _submit_phase5(self, target_qubit: str) -> List[str]:
        """Submit Phase 5: Final DRAG vs Gaussian validation."""
        from qiskit import QuantumCircuit

        job_ids = []

        # Final validation circuits
        for pulse_type in ["drag_final", "gaussian_final"]:
            qc = QuantumCircuit(1, 1)
            qc.x(0)
            qc.measure(0, 0)

            job_id = self.job_manager.submit_job(
                circuit=qc,
                shots=2048,
                metadata={
                    "phase": 5,
                    "experiment": pulse_type,
                    "qubit": target_qubit,
                    "final_validation": True,
                },
            )
            job_ids.append(job_id)

        return job_ids

    def _get_session_duration(self) -> str:
        """Calculate session duration."""
        if not self.job_manager.jobs:
            return "N/A"

        start_times = [job.submit_time for job in self.job_manager.jobs.values()]
        end_times = [
            job.completion_time
            for job in self.job_manager.jobs.values()
            if job.completion_time
        ]

        if not start_times or not end_times:
            return "In progress"

        duration_sec = max(end_times) - min(start_times)
        hours = int(duration_sec // 3600)
        minutes = int((duration_sec % 3600) // 60)

        return f"{hours}h {minutes}m"


def main():
    """Main entry point for async hardware validation."""
    parser = argparse.ArgumentParser(
        description="IQM Hardware Validation - Async Job Submission"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on emulator instead of real hardware",
    )
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="Submit jobs and exit (don't wait for completion)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from saved session file",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Check status only (requires --resume)",
    )
    parser.add_argument(
        "--target-qubit",
        type=str,
        default="QB1",
        help="Target qubit for validation (default: QB1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Maximum wait time in seconds",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.INFO)

    logger.info("=" * 80)
    logger.info("IQM HARDWARE VALIDATION - ASYNC JOB SUBMISSION")
    logger.info("=" * 80)

    # Resume existing session
    if args.resume:
        logger.info(f"\nRestoring session from: {args.resume}")
        validator = AsyncHardwareValidator.load_session(Path(args.resume))

        # Status check only
        if args.status_only:
            summary = validator.get_status_summary()
            logger.info("\n" + json.dumps(summary, indent=2))
            return

    # Start new session
    else:
        validator = AsyncHardwareValidator(use_emulator=args.dry_run)

        logger.info(f"\nSession ID: {validator.session_id}")
        logger.info(f"Mode: {'EMULATOR' if args.dry_run else 'HARDWARE'}")

        # Submit all jobs
        logger.info("\nSubmitting batch jobs...")
        submission_summary = validator.submit_all_phases(target_qubit=args.target_qubit)

        logger.info("\n" + json.dumps(submission_summary, indent=2))

        # Save session after submission
        session_file = validator.save_session()
        logger.info(f"\nSession saved: {session_file}")

        # If submit-only, exit here
        if args.submit_only:
            logger.info("\n✓ Jobs submitted successfully!")
            logger.info("You can disconnect now - jobs are running on hardware")
            logger.info(f"\nTo resume: python {sys.argv[0]} --resume {session_file}")
            return

    # Wait for completion and collect results
    logger.info("\nWaiting for job completion...")
    results = validator.wait_and_collect_results(timeout=args.timeout)

    # Generate report
    logger.info("\nGenerating report...")
    report_path = validator.generate_report(results)

    # Final save
    validator.save_session()

    logger.info("\n" + "=" * 80)
    logger.info("✓ HARDWARE VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nReport: {report_path}")
    logger.info(f"Session: {validator.session_id}")
    logger.info(f"Total Credits: {validator.job_manager.total_credits_used}")


if __name__ == "__main__":
    main()
