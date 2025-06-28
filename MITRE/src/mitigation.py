from typing import Sequence

import numpy as np

# For circuit folding:
from qiskit.circuit import QuantumCircuit
from qiskit.result import QuasiDistribution
from qiskit_experiments.library.characterization import LocalReadoutError

__all__ = ["ReadoutCal", "ZNE", "mitigate"]


class ReadoutCal:
    """Read-out error calibration using Qiskit-Experiments LocalReadoutError."""

    def __init__(self, qubits, backend, shots: int = 4_096):
        self.qubits = tuple(qubits)
        self.backend = backend
        self.shots = shots
        self._mitigator = None  # type: LocalReadoutMitigator | None

    # ---------- public API ----------
    def fit(self):
        """Characterize read-out error and cache a LocalReadoutMitigator."""
        exp = LocalReadoutError(physical_qubits=self.qubits, backend=self.backend)
        exp.set_run_options(shots=self.shots)  # ✔ correct place
        edata = exp.run().block_for_results()  # backend already set
        # first analysis result is always the mitigator
        self._mitigator = edata.analysis_results(0).value

    def apply(self, raw_counts) -> QuasiDistribution:
        if self._mitigator is None:
            raise RuntimeError("Call .fit() before .apply()")
        return self._mitigator.quasi_probabilities(raw_counts)


class ZNE:
    """Zero-noise extrapolation via gate folding + Richardson polynomial fit."""

    scale_factors = (1, 3, 5)

    def fold(self, circuit: QuantumCircuit, scale: int) -> QuantumCircuit:
        if scale == 1:
            return circuit.copy()
        folded = QuantumCircuit(circuit.num_qubits)
        for instr, qargs, cargs in circuit.data:
            folded.append(instr, qargs, cargs)
            if instr.name != "measure":
                for _ in range((scale - 1) // 2):
                    folded.append(instr, qargs, cargs)
                    folded.append(instr.inverse(), qargs, cargs)
        return folded

    def extrapolate(self, noisy_values: Sequence[float]) -> float:
        # Fit polynomial through (scale, noisy_value) points and evaluate at 0
        coeffs = np.polyfit(self.scale_factors, noisy_values, deg=len(noisy_values) - 1)
        return float(np.polyval(coeffs, 0.0))


# --- Convenience function ---
def mitigate(counts, cal: ReadoutCal, zne: ZNE) -> float:
    qpd = cal.apply(counts)
    exp0 = qpd.get(0, 0) + qpd.get(3, 0) - qpd.get(1, 0) - qpd.get(2, 0)
    noisy = []
    for s in zne.scale_factors:
        noisy.append(
            exp0
        )  # Replace this stub with actual folded evaluation in real use
    return zne.extrapolate(noisy)
