# R1.10-OATH - Phase 3

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=<YOUR GIT .git LINK>)

---

## Project Purpose

This project explores quantum chemistry simulations for asphalt-like molecules, specifically focusing on the electronic structure of Dibenzothiophene (DBT).

### Phase 3 Objectives

-   Establish a reproducible, version-controlled workflow.
-   Implement a robust CI/CD pipeline for automated testing and validation.
-   Refine and document the process for generating molecular Hamiltonians and running VQE simulations.

## Environment Setup

To create and activate the Conda environment for this project, run the following commands:

```bash
conda env create -f environment.yml
conda activate asphalt-qchem-p3
```

## Reproduce Results

To reproduce the core results of this project, run the main experiment script (to be created):

```bash
# Example command
python src/run_experiment.py
```
