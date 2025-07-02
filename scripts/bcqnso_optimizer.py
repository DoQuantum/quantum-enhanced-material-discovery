import pennylane as qml
from pennylane import numpy as np

class BCQNSO:
    """
    Block-Coordinate Quantum Natural Surrogate Optimizer (BCQNSO)
    for k-UpCCGSD ansatz, with debug printouts.
    """
    def __init__(self, blocks, init_shots=1024, lr=0.1, reg_lambda=1e-3,
                 min_shots=128, max_shots=8192, grad_thresh=1e-2):
        self.blocks = blocks
        self.shots = init_shots
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.min_shots = min_shots
        self.max_shots = max_shots
        self.grad_thresh = grad_thresh
        print(f"[BCQNSO __init__] blocks={blocks}, shots={self.shots}, lr={self.lr}, "
              f"reg_lambda={self.reg_lambda}, grad_thresh={self.grad_thresh}")

    def step(self, cost_fn, params):
        """
        Perform one optimization step over all blocks, with debug prints.
        """
        print("\n[BCQNSO step] Starting step")
        print("[BCQNSO step] Current shots:", self.shots)
        new_params = params.copy()

        # Prepare metric and gradient functions
        fisher_fn = qml.metric_tensor(cost_fn)
        grad_fn = qml.grad(cost_fn, argnum=0)

        # Compute full gradient once
        grads = grad_fn(new_params)
        print("[BCQNSO step] Full gradient vector:", grads)

        # Compute full Fisher metric once
        full_fisher = fisher_fn(new_params)
        print("[BCQNSO step] Full Fisher matrix shape:", full_fisher.shape)

        for i, block in enumerate(self.blocks):
            # Determine block indices
            if isinstance(block, slice):
                idx = list(range(block.start, block.stop))
            else:
                idx = list(block)
            print(f"\n[BCQNSO step] Block {i}: indices {idx}")

            # Extract block gradient
            grad_block = grads[idx]
            print(f"[BCQNSO step] grad_block:", grad_block)

            # Extract Fisher submatrix & compute diagonal Hessian surrogate
            sub_fisher = full_fisher[np.ix_(idx, idx)]
            hess_diag = np.diag(sub_fisher) + self.reg_lambda
            print(f"[BCQNSO step] sub_fisher shape: {sub_fisher.shape}")
            print(f"[BCQNSO step] hess_diag:", hess_diag)

            # Compute block update
            update = - self.lr * grad_block / hess_diag
            print(f"[BCQNSO step] update for block:", update)

            # Apply update
            new_params[idx] += update
            print(f"[BCQNSO step] new_params[{idx}] after update:", new_params[idx])

            # Adapt shots if using a sampled device
            if hasattr(cost_fn, 'device') and cost_fn.device.shots is not None:
                new_shots = self._update_shots(grad_block)
                print(f"[BCQNSO step] Adapting shots: {self.shots} -> {new_shots}")
                self.shots = new_shots
                cost_fn.device.shots = int(self.shots)
                print(f"[BCQNSO step] cost_fn.device.shots set to {cost_fn.device.shots}")

        print("\n[BCQNSO step] Step complete, returning new_params")
        return new_params

    def _update_shots(self, grad_block):
        """
        Increase or decrease shot count based on gradient L2 norm, with debug.
        """
        norm = np.linalg.norm(grad_block)
        print(f"[BCQNSO _update_shots] grad_block norm: {norm}")
        if norm > self.grad_thresh:
            new = min(self.shots * 2, self.max_shots)
            print(f"[BCQNSO _update_shots] Norm above threshold, increasing shots to {new}")
        else:
            new = max(self.shots // 2, self.min_shots)
            print(f"[BCQNSO _update_shots] Norm below threshold, decreasing shots to {new}")
        return new
