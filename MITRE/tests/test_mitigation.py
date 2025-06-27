import unittest
from qiskit.circuit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error 
from src.mitigation import ReadoutCal, ZNE
import numpy as np
from qiskit import transpile, assemble, execute 


class TestMitigation(unittest.TestCase):

    def test_readout_calibration_reduces_bias(self):
        """
        Build a 2-qubit Bell circuit.
        Run with artificial readout errors (simulate noise model).
        Verify that ReadoutCal reduces bias in measurement results.
        """
        backend = Aer.get_backend('qasm_simulator')
        
        # Define a noise model that flips measurement results with some probability
        noise_model = NoiseModel()
        error = depolarizing_error(0.05, 1)  # 5% depolarizing on single qubit (approx readout error)
        noise_model.add_all_qubit_quantum_error(error, ['measure'])
        
        # Prepare Bell state circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0,1], [0,1])
        
        # Run circuit with noise to get raw counts
        job = execute(qc, backend=backend, shots=8192, noise_model=noise_model)
        noisy_counts = job.result().get_counts()

        # Readout Calibration on noisy backend (simulate ideal backend here)
        readout = ReadoutCal(backend=backend, qubits=[0,1], shots=8192)
        readout.fit()
        
        # Apply readout calibration on noisy counts
        mitigated_probs = readout.apply(noisy_counts)

        # Calculate expectation values before and after calibration
        def expectation(counts):
            shots = sum(counts.values())
            exp_val = 0
            for bitstring, count in counts.items():
                parity = (-1) ** (bitstring.count('1') % 2)
                exp_val += parity * count / shots
            return exp_val

        noisy_exp = expectation(noisy_counts)
        mitigated_exp = 0
        # mitigated_probs is quasi-probabilities (dict with floats)
        for bitstring, prob in mitigated_probs.items():
            parity = (-1) ** (bitstring.count('1') % 2)
            mitigated_exp += parity * prob

        # Ideal Bell state expectation is 1 for ZZ parity measurement
        # We expect mitigation to improve (closer to 1) than raw noisy result
        self.assertTrue(abs(mitigated_exp - 1) < abs(noisy_exp - 1), 
                        msg="Readout calibration did not reduce bias.")

    def test_zne_recovers_expectation(self):
        """
        Fold a 1-qubit circuit with depolarizing noise injected.
        Check that ZNE recovers the true expectation value within tolerance.
        """
        backend = Aer.get_backend('qasm_simulator')
        
        # 1-qubit circuit with H gate and measurement
        qc = QuantumCircuit(1,1)
        qc.h(0)
        qc.measure(0,0)
        
        # Define depolarizing noise model
        noise_model = NoiseModel()
        error = depolarizing_error(0.1, 1)  # 10% depolarizing noise
        noise_model.add_all_qubit_quantum_error(error, ['u3'])  # Add noise on all single qubit gates
        
        # ZNE instance with noise backend
        zne = ZNE(backend=backend, shots=8192)
        
        # Mitigate expectation value
        mitigated_exp = zne.mitigate(qc)

        # Ideal expectation for H state on Z basis is 0 (equal superposition)
        # With noise, expect deviation, ZNE tries to recover 0
        self.assertAlmostEqual(mitigated_exp, 0, delta=0.2, 
                               msg="ZNE did not recover expectation close to ideal.")

if __name__ == "__main__":
    unittest.main()
