"""Quantum/classical model utilities for a dibenzothiophene example.

This module provides a minimal framework for exploring potential quantum
advantage on a toy dataset derived from the ``dibenzothiophene.xyz`` geometry
included in this repository.  It exposes helper functions to build a simple
variational quantum classifier, a classical baseline model and evaluation
utilities.  The ``__main__`` section demonstrates how to generate a synthetic
dataset from the molecule coordinates and train both models.
"""

from pathlib import Path

import pennylane as qml
from pennylane import numpy as np


def create_quantum_circuit(n_qubits: int, n_layers: int):
    """Return a QNode implementing a variational quantum circuit."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
    return circuit, weight_shape


def train_quantum_classifier(
    data: np.ndarray,
    labels: np.ndarray,
    n_qubits: int | None = None,
    n_layers: int = 2,
    steps: int = 50,
) -> tuple[qml.QNode, np.ndarray]:
    """Train a simple quantum classifier on the provided data."""
    if n_qubits is None:
        n_qubits = data.shape[1]

    qnode, weight_shape = create_quantum_circuit(n_qubits, n_layers)
    weights = np.random.randn(*weight_shape, requires_grad=True)
    opt = qml.AdamOptimizer(0.01)

    for _ in range(steps):
        for x, y in zip(data, labels):
            def loss(w):
                pred = qnode(x, w)
                return (pred - y) ** 2

            weights = opt.step(loss, weights)

    return qnode, weights


def load_dbt_coordinates(xyz_file: Path | str | None = None) -> np.ndarray:
    """Return flattened atomic coordinates from the XYZ file."""
    if xyz_file is None:
        xyz_path = Path(__file__).resolve().parent / "dibenzothiophene.xyz"
    else:
        xyz_path = Path(xyz_file)
    if not xyz_path.is_file():
        raise FileNotFoundError(f"{xyz_path} not found")

    lines = xyz_path.read_text().splitlines()[2:]
    coords = []
    for line in lines:
        parts = line.split()
        coords.extend([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords, dtype=float)


def synthetic_dbt_dataset(
    num_samples: int = 20, noise_std: float = 0.05, n_features: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy dataset using dibenzothiophene coordinates."""
    base = load_dbt_coordinates()[:n_features]
    data = np.tile(base, (num_samples, 1))
    data += noise_std * np.random.randn(*data.shape)

    thresh = np.sum(base)
    labels = (np.sum(data, axis=1) > thresh).astype(int)
    return data, labels


def classical_baseline(data, labels):
    """Train a logistic regression model as a classical benchmark."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for the classical baseline"
        ) from exc

    model = LogisticRegression(max_iter=500)
    model.fit(data, labels)
    return model


def evaluate_models(qnode, q_weights, classical_model, test_data, test_labels):
    """Return mean squared errors for the quantum and classical models."""
    q_errors = []
    for x, y in zip(test_data, test_labels):
        pred = qnode(x, q_weights)
        q_errors.append((pred - y) ** 2)
    quantum_mse = np.mean(q_errors)

    classical_pred = classical_model.predict(test_data)
    classical_mse = ((classical_pred - test_labels) ** 2).mean()

    return quantum_mse, classical_mse


if __name__ == "__main__":
    # Demonstration using the dibenzothiophene-based toy dataset
    np.random.seed(0)
    data, labels = synthetic_dbt_dataset(num_samples=20)

    q_model, q_weights = train_quantum_classifier(data, labels)
    clf_model = classical_baseline(data, labels)
    qmse, cmse = evaluate_models(q_model, q_weights, clf_model, data, labels)
    print(f"Quantum MSE: {qmse:.4f}\nClassical MSE: {cmse:.4f}")
