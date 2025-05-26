import pennylane as qml
from pennylane import numpy as pnp # Use PennyLane's wrapped numpy
import numpy as np # Standard numpy for data generation and final output type

# --- PennyLane Feature Map Definition ---
def pennylane_scalar_feature_map_ops(features, num_wires):
    """
    Applies PennyLane operations for a scalar feature map.
    This function is called by QNodes or other PennyLane constructs.
    """
    if len(features) != num_wires:
        raise ValueError(f"Number of features ({len(features)}) must match number of wires ({num_wires}).")
    for i in range(num_wires):
        qml.Hadamard(wires=i)
    for i in range(num_wires):
        qml.RY(features[i], wires=i)
    if num_wires > 1:
        for i in range(num_wires - 1):
            qml.CNOT(wires=[i, i + 1])

# --- PennyLane Quantum Kernel Computation ---
def compute_pennylane_quantum_kernel_matrix(x_data, y_data=None, num_features=None,
                                            device_name="default.qubit", shots=None):
    if num_features is None:
        num_features = x_data.shape[1]

    # Ensure data is PennyLane numpy array, without gradients for kernel inputs
    x1_pnp = pnp.array(x_data, requires_grad=False)
    
    print_msg_suffix = ""
    if y_data is None:
        x2_pnp = x1_pnp # Kernel of x_data with itself
        print_msg_suffix = " (kernel of x_data with itself)"
    else:
        x2_pnp = pnp.array(y_data, requires_grad=False)
        print_msg_suffix = f", y_data shape: {x2_pnp.shape}"
        
    dev = qml.device(device_name, wires=num_features, shots=shots)

    # This is the subroutine that defines the unitary U(params) from the feature map
    def feature_map_unitary_subroutine(params):
        pennylane_scalar_feature_map_ops(params, num_wires=num_features)

    @qml.qnode(dev, interface="autograd")
    def individual_kernel_value_qnode(params_i, params_j):
        feature_map_unitary_subroutine(params_j)
        qml.adjoint(feature_map_unitary_subroutine)(params_i)
        return qml.expval(
            qml.Projector([0] * num_features, wires=range(num_features))
        )


    print(f"Computing PennyLane kernel matrix. x_data shape: {x1_pnp.shape}{print_msg_suffix}")
        
    # qml.kernels.kernel_matrix will iterate, calling individual_kernel_value_qnode(x_i, x_j)
    # for each pair from X1 and X2.
    kernel_matrix_result = qml.kernels.kernel_matrix(
        X1=x1_pnp,
        X2=x2_pnp,
        kernel=individual_kernel_value_qnode 
    )

    print("PennyLane Kernel matrix computation complete.")
    # Ensure standard numpy array for broader compatibility downstream
    if hasattr(kernel_matrix_result, 'numpy'): # For PennyLane tensors
        return kernel_matrix_result.numpy()
    return np.array(kernel_matrix_result) # For other cases, ensure it's a np.array


if __name__ == '__main__':
    print("--- PennyLane Quantum Utilities Example ---")
    
    num_scalar_features = 3
    
    print("\n--- PennyLane Quantum Kernel Computation Example ---")
    # import numpy as np # Already imported at the top
    np.random.seed(42)
    num_train_samples = 5
    num_test_samples = 3
    
    X_train_example_np = np.random.rand(num_train_samples, num_scalar_features).astype(np.float64) * np.pi
    X_test_example_np = np.random.rand(num_test_samples, num_scalar_features).astype(np.float64) * np.pi

    print(f"\nDummy X_train_example (first 2 samples):\n{X_train_example_np[:2]}")
    print(f"Dummy X_test_example (first 2 samples):\n{X_test_example_np[:2]}")

    print("\nCalculating PennyLane training kernel matrix (K_train_train):")
    kernel_train_train = compute_pennylane_quantum_kernel_matrix(
        x_data=X_train_example_np,
        num_features=num_scalar_features
    )
    print(f"PennyLane Training kernel matrix shape: {kernel_train_train.shape}")
    print(f"PennyLane Training kernel matrix (K_train_train):\n{kernel_train_train}")

    print("\nCalculating PennyLane testing kernel matrix (K_test_train):")
    kernel_test_train = compute_pennylane_quantum_kernel_matrix(
        x_data=X_test_example_np,
        y_data=X_train_example_np,
        num_features=num_scalar_features
    )
    print(f"PennyLane Testing kernel matrix shape: {kernel_test_train.shape}")
    print(f"PennyLane Testing kernel matrix (K_test_train):\n{kernel_test_train}")