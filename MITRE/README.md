# R1.10-OATH - Phase 3

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/DoQuantum/r1.10-oath.git&branch=phase3)

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

## Noise Mitigation

Combine measurement error calibration and zero-noise extrapolation:

```python
from src.mitigation import ReadoutCal, ZNE, mitigate
from qiskit_ibm_runtime.fake_provider import FakePerth
from qiskit import QuantumCircuit

backend = FakePerth()
bell = QuantumCircuit(2)
bell.h(0); bell.cx(0, 1); bell.measure_all()

cal = ReadoutCal([0, 1], backend, shots=8192)
cal.fit()

raw_counts = backend.run(bell).result().get_counts()
zne = ZNE()
zz = mitigate(raw_counts, cal, zne)

print("Mitigated ⟨Z⊗Z⟩:", zz)
```

