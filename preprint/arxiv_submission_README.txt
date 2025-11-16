arXiv Submission Package for QubitPulseOpt
==========================================

File: arxiv_submission_malarchick_qubitpulseopt.tar.gz

Contents:
- preprint.tex (main LaTeX source)
- figures/ directory with 5 PNG images:
  * bloch_trajectory.png
  * architecture_workflow.png
  * verified_fidelity_convergence.png
  * verified_pulse_comparison.png
  * verified_error_comparison.png

Package created/updated: 2025-11-16
Updated test suite metrics: 864 tests, 74% coverage
All figures verified to display correctly in PDF
Fixed figure float placement (using [h!] specifiers)

Key Updates in This Version:
- Updated test suite: 659 tests (59%) -> 864 tests (74%)
- Added 85%+ coverage of hardware integration modules
- Added AI assistance disclosure (ethical transparency)
- Emphasized mocked API testing for reproducibility

To verify package:
  tar -tzf arxiv_submission_malarchick_qubitpulseopt.tar.gz

To extract and compile:
  tar -xzf arxiv_submission_malarchick_qubitpulseopt.tar.gz
  pdflatex preprint.tex
  pdflatex preprint.tex

Output: 11-page PDF, ~1.4 MB

Submission type: Replacement/update
Previous version: 59% test coverage (659 tests), no AI disclosure
New version: 74% test coverage (864 tests), includes AI disclosure
