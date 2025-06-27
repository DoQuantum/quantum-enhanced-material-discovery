from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit import execute
from mitiq import zne
from mitiq.interface.mitiq_qiskit import convert_from_qiskit, convert_to_qiskit

class ReadoutCal:
    """
    Readout Calibration mitigation class.
    Builds calibration matrices and applies readout error correction.
    """
    def __init__(self, backend, qubits, shots=8192):
        """
        Initialize with backend and qubits to calibrate.
        
        :param backend: Qiskit backend (simulator or real device)
        :param qubits: list of qubit indices to calibrate
        :param shots: number of shots for calibration circuits
        """
        self.backend = backend
        self.qubits = qubits
        self.shots = shots
        self.meas_fitter = None

    def fit(self):
        """
        Generate calibration circuits, run on backend, and create fitter.
        """
        cal_circuits, state_labels = complete_meas_cal(qubit_list=self.qubits, circlabel='mcal')
        job = execute(cal_circuits, backend=self.backend, shots=self.shots)
        results = job.result()
        self.meas_fitter = CompleteMeasFitter(results, state_labels, circlabel='mcal')
        print("Readout calibration matrices fitted.")

    def apply(self, counts):
        """
        Apply readout error mitigation on raw counts.
        
        :param counts: dict, raw counts from circuit execution
        :return: corrected quasi-probabilities as dict
        """
        if self.meas_fitter is None:
            raise RuntimeError("Calibration fitter is not fitted. Call fit() first.")
        mitigated_probs = self.meas_fitter.filter.apply(counts)
        return mitigated_probs


class ZNE:
    """
    Zero Noise Extrapolation (ZNE) mitigation class.
    Uses folding strategy (scale factors 1, 3, 5) + Richardson extrapolation.
    """
    def __init__(self, backend, shots=8192, scale_factors=[1, 3, 5]):
        self.backend = backend
        self.shots = shots
        self.scale_factors = scale_factors

    def _execute_circuit(self, circ):
        """
        Helper to execute a single circuit and return expectation value of Z^{\otimes n}.
        """
        job = execute(circ, backend=self.backend, shots=self.shots)
        counts = job.result().get_counts()
        return self._expectation_from_counts(counts)

    def _expectation_from_counts(self, counts):
        """
        Compute expectation value assuming parity measurement (Z^{\otimes n}).
        """
        shots = sum(counts.values())
        exp_val = 0
        for bitstring, count in counts.items():
            parity = (-1) ** (bitstring.count('1') % 2)  # parity of ones
            exp_val += parity * count / shots
        return exp_val

    def mitigate(self, circuit):
        """
        Apply zero noise extrapolation to estimate zero-noise expectation value.
        
        :param circuit: QuantumCircuit to run with noise scaling
        :return: mitigated expectation value (float)
        """
        noisy_circ = convert_from_qiskit(circuit)

        def executor(circ):
            qiskit_circ = convert_to_qiskit(circ)
            return self._execute_circuit(qiskit_circ)

        # Use mitiq's execute_with_zne utility with RichardsonFactory
        exp_val = zne.execute_with_zne(
            noisy_circ,
            executor,
            scale_factors=self.scale_factors
        )
        return exp_val


def mitigate(counts, circuit, cal: ReadoutCal, zne_obj: ZNE):
    """
    Convenience function that applies readout calibration first, then ZNE on a circuit.
    
    :param counts: raw counts dict from executing circuit once
    :param circuit: QuantumCircuit object corresponding to counts
    :param cal: fitted ReadoutCal object
    :param zne_obj: initialized ZNE object
    :return: mitigated expectation value (float)
    """
    # Apply readout calibration
    calibrated_probs = cal.apply(counts)

    # Convert calibrated probs to expectation value
    shots = sum(calibrated_probs.values())
    exp_val = 0
    for bitstring, prob in calibrated_probs.items():
        parity = (-1) ** (bitstring.count('1') % 2)
        exp_val += parity * prob  # prob here is quasi-probability

    # Now apply ZNE mitigation on the circuit
    zne_val = zne_obj.mitigate(circuit)

    # Return ZNE corrected expectation value (can choose to combine or return zne_val)
    return zne_val
