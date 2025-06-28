# tests/test_mitigation.py
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

import pytest
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime.fake_provider import FakePerth

from src.mitigation import ZNE, ReadoutCal, mitigate  # ← added mitigate import


def bell_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def test_readout_calibration_reduces_bias():
    backend = FakePerth()
    raw_counts = backend.run(bell_circuit()).result().get_counts()

    cal = ReadoutCal([0, 1], backend, shots=4096)
    cal.fit()
    mitigated = cal.apply(raw_counts)

    p_raw = (raw_counts.get("00", 0) + raw_counts.get("11", 0)) / sum(raw_counts.values())
    p_mit = mitigated.get(0, 0) + mitigated.get(3, 0)
    assert p_mit > p_raw, "Calibration did not improve Bell-state parity"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_zne_richardson_extrapolation():
    estimator = StatevectorEstimator()
    folded = bell_circuit()
    folded.remove_final_measurements()

    observable = Pauli("ZZ")
    zne = ZNE()

    noisy_vals = []
    for scale in zne.scale_factors:
        circ = zne.fold(folded, scale)
        job = estimator.run([(circ, observable)])
        result = job.result()
        pr = result[0]
        evs = pr.data.evs
        ev = evs.item() if hasattr(evs, "item") else evs
        noisy_vals.append(ev)

    ezero = zne.extrapolate(noisy_vals)
    assert np.isclose(ezero, 1.0, atol=0.05), f"ZNE failed: {ezero}"


def test_full_mitigation_workflow():
    backend = FakePerth()
    cal = ReadoutCal([0, 1], backend, shots=4096)
    cal.fit()
    zne = ZNE()
    circ = bell_circuit()

    zz = mitigate(circ, cal, zne, backend)  # now using circuit + backend
    print("DEBUG mitigated ⟨ZZ⟩:", zz)
    assert isinstance(zz, float)
    assert -1.05 <= zz <= 1.05, f"Mitigated expectation {zz:.3f} out of allowed bounds"

