"""
vqsvd_ideal_simulator.py

Full pipeline for Phase 2/3 of the EM-VQSVD project:

    1. Compute classical SVD of a test matrix M (ground truth).
    2. Build a parameterized quantum circuit ansatz (two of them: U(theta), V(phi)).
    3. Train them using the Ky Fan theorem loss function.
    4. Run everything on an IDEAL (noiseless) quantum simulator -- PennyLane's
       'default.qubit' device, using exact statevector simulation.
    5. Compare the learned singular values/vectors against the classical
       ground truth.

Requires: pennylane, numpy   (pip install pennylane)
"""

import pennylane as qml
from pennylane import numpy as np   # autograd-wrapped numpy -> differentiable

np.random.seed(42)

# ----------------------------------------------------------------------
# 0. Problem setup: matrix, qubits, number of singular values wanted
# ----------------------------------------------------------------------
n_qubits = 2                 # 2 qubits -> 4x4 matrix
dim = 2 ** n_qubits          # 4
T = dim                      # how many singular values/vectors to extract (all 4 here)
n_layers = 3                 # ansatz depth

# A fixed, real 4x4 test matrix (replace with your own matrix, or an image
# patch, or a neural-network weight slice, as your project's "real" test case)
M = np.array([
    [0.9, 0.3, 0.1, 0.4],
    [0.2, 0.8, 0.5, 0.1],
    [0.3, 0.2, 0.7, 0.6],
    [0.5, 0.1, 0.2, 0.9]
])

# ----------------------------------------------------------------------
# 1. Classical SVD -- ground truth
# ----------------------------------------------------------------------
U_true, S_true, Vh_true = np.linalg.svd(M)
print("=" * 65)
print("STEP 1 -- Classical SVD (ground truth, computed with NumPy)")
print("=" * 65)
print("Singular values (sorted, descending):")
print("  ", np.round(S_true, 4))
print()

# ----------------------------------------------------------------------
# 2. Ansatz: hardware-efficient parameterized circuit
#    Each layer = RY + RZ rotations on every qubit, then a ring of CNOTs.
# ----------------------------------------------------------------------
def ansatz(params, n_qubits, n_layers):
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.RY(params[l, q, 0], wires=q)
            qml.RZ(params[l, q, 1], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])


dev = qml.device("default.qubit", wires=n_qubits)   # IDEAL, noiseless simulator


@qml.qnode(dev, interface="autograd", diff_method="backprop")
def get_state(params, basis_index):
    """Prepare computational basis state |basis_index>, apply the ansatz,
    and return the resulting statevector -- exactly (no shot noise)."""
    qml.BasisState(np.array(basis_index), wires=range(n_qubits))
    ansatz(params, n_qubits, n_layers)
    return qml.state()


def int_to_bits(i, n):
    return [int(b) for b in format(i, f"0{n}b")]


# Precompute the basis-state bit patterns once
basis_patterns = [int_to_bits(i, n_qubits) for i in range(dim)]

# ----------------------------------------------------------------------
# 3. Ky Fan loss:  L = sum_i  q_i * Re( <psi_i| U(theta)^dagger M V(phi) |psi_i> )
#    Maximizing this (with descending weights q_i) forces the trained
#    circuits to output the true singular vectors, in the correct order,
#    and the loss terms themselves converge to the true singular values.
# ----------------------------------------------------------------------
weights_q = np.array([float(T - i) for i in range(T)])   # e.g. [4, 3, 2, 1]


def ky_fan_loss(theta, phi):
    total = 0.0
    for i in range(T):
        u_i = get_state(theta, basis_patterns[i])   # U(theta)|i>
        v_i = get_state(phi, basis_patterns[i])      # V(phi)|i>
        term = np.real(np.sum(np.conj(u_i) * (M @ v_i)))   # Re( u_i^dagger M v_i )
        total = total + weights_q[i] * term
    return total


def cost(theta, phi):
    return -ky_fan_loss(theta, phi)   # minimize negative loss = maximize loss


# ----------------------------------------------------------------------
# 4. Train on the ideal simulator
# ----------------------------------------------------------------------
theta = np.array(np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 2)), requires_grad=True)
phi   = np.array(np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 2)), requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.1)
n_iterations = 150

print("=" * 65)
print("STEP 2 -- Training VQSVD (ansatz + Ky Fan loss) on IDEAL simulator")
print("=" * 65)
for it in range(n_iterations):
    (theta, phi), loss_val = opt.step_and_cost(lambda t, p: cost(t, p), theta, phi)
    if (it + 1) % 30 == 0 or it == 0:
        print(f"  iteration {it+1:4d}  |  loss = {-loss_val:.4f}")

print()

# ----------------------------------------------------------------------
# 5. Extract learned singular values / vectors, compare to ground truth
# ----------------------------------------------------------------------
learned_sigmas = []
learned_U_cols = []
learned_V_cols = []
for i in range(T):
    u_i = get_state(theta, basis_patterns[i])
    v_i = get_state(phi, basis_patterns[i])
    sigma_i = np.real(np.sum(np.conj(u_i) * (M @ v_i)))
    learned_sigmas.append(float(sigma_i))
    learned_U_cols.append(u_i)
    learned_V_cols.append(v_i)

learned_sigmas = np.array(learned_sigmas)

print("=" * 65)
print("STEP 3 -- Comparison: classical SVD vs. trained VQSVD (ideal simulator)")
print("=" * 65)
print(f"{'Index':<8}{'Classical sigma':<20}{'VQSVD sigma':<20}{'Abs. error':<12}")
for i in range(T):
    err = abs(S_true[i] - learned_sigmas[i])
    print(f"{i:<8}{S_true[i]:<20.4f}{learned_sigmas[i]:<20.4f}{err:<12.4f}")

overall_error = np.mean(np.abs(S_true - np.sort(learned_sigmas)[::-1]))
print(f"\nMean absolute error across all singular values: {overall_error:.4f}")

# Reconstruction check: how well does the learned decomposition
# reconstruct M compared to the true one (should both be ~exact here
# since T = full rank = dim)
U_learned = np.stack(learned_U_cols, axis=1)   # columns = u_i
V_learned = np.stack(learned_V_cols, axis=1)   # columns = v_i
M_reconstructed = np.real(U_learned @ np.diag(learned_sigmas) @ np.conj(V_learned).T)

recon_error = np.linalg.norm(M - M_reconstructed) / np.linalg.norm(M)
print(f"Relative Frobenius reconstruction error (VQSVD vs. true M): {recon_error:.4f}")