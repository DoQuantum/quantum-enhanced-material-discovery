from typing import Sequence

import numpy as np

# For circuit folding:
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.result import QuasiDistribution
from qiskit_experiments.library.characterization import LocalReadoutError
from qiskit_ibm_runtime import SamplerV2

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
    scale_factors = (1, 3, 5)

    def fold(self, circuit: QuantumCircuit, scale: int) -> QuantumCircuit:
        if scale == 1:
            return circuit.copy()
        # Create new circuit inheriting registers
        folded = circuit.copy_empty_like()
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
def mitigate(original_circ, cal: ReadoutCal, zne: ZNE, backend) -> float:
    # Make a clean base circuit with explicit measurement
    base = original_circ.copy()
    base.remove_final_measurements(inplace=True)  # remove any previous ones
    base.measure_all()                           # add fresh ones
    sampler = SamplerV2(mode=backend)
    noisy = []

    for scale in zne.scale_factors:
        circ_s = zne.fold(base, scale)
        circ_t = transpile(circ_s, backend=backend)

        job = sampler.run([circ_t], shots=cal.shots)
        pub = job.result()[0]

        # 🔍 Use .data.meas only—it will exist since we added measurements
        counts = pub.data.meas.get_counts()
        qpd = cal.apply(counts)

        exp = qpd.get(0, 0) + qpd.get(3, 0) - qpd.get(1, 0) - qpd.get(2, 0)
        noisy.append(exp)

    return zne.extrapolate(noisy)