"""
vqsvd_noisy_eval.py

PART 2 of the noisy-VQSVD pipeline. Loads the ansatz trained by
vqsvd_train_symmetric.py (on the ideal simulator) and the 16x16 test
matrix M, then evaluates <psi_i|M|psi_i> for the top-T learned singular
vectors under FIVE conditions:

    1. Ideal simulator (no noise)               -- sanity check, matches Part 1
    2. Noisy simulator, no mitigation            -- realistic NISQ-style noise
    3. Noisy + readout error mitigation only
    4. Noisy + Zero-Noise Extrapolation (ZNE) only
    5. Noisy + ZNE AND readout mitigation (combined)

HOW <psi|M|psi> IS MEASURED FOR A MULTI-QUBIT M:
M was built (in Part 1) as a weighted sum of Pauli strings, e.g.
M = sum_k c_k * (P_k1 (x) P_k2 (x) ... (x) P_kn), P in {I, X, Z}.
For a Hermitian sum-of-Paulis, <psi|M|psi> = sum_k c_k * <psi|P_k|psi>,
and each Pauli-string expectation is measured with a real quantum circuit:
rotate any qubit with an X factor into the Z basis (apply H), then measure
every qubit in the computational (Z) basis. This is the same principle
used in svd_error_mitigation_proof.py, generalized to many qubits.

NOISE MODEL:
- Gate noise: a DepolarizingChannel after every single-qubit gate (rate p1)
  and after every qubit touched by a CNOT (rate p2, applied per qubit as
  an approximation of true 2-qubit depolarizing noise).
- Readout noise: a BitFlip channel (rate p_read) on every qubit, applied
  once, right before the final measurement -- independent of fold count.

MITIGATION:
- ZNE: "unitary folding" of the WHOLE ansatz (apply ansatz, then
  ansatz^-1 + ansatz repeated), amplifying gate noise at scale factors
  1x, 3x, 5x; linear extrapolation back to the 0x (zero-noise) point.
- Readout mitigation: for INDEPENDENT per-qubit bit-flip readout noise,
  a Pauli-string expectation value is damped by a known factor
  (1 - 2*p_read) for every qubit the string acts on non-trivially.
  Dividing the measured value by this factor is an exact analytic
  correction (calibrated by measuring p_read directly), avoiding the
  need for a full 2^n x 2^n confusion matrix.

Requires: pennylane, numpy
"""

import time
import numpy as onp
import pennylane as qml

# ----------------------------------------------------------------------
# 0. Load the trained ansatz and problem data from Part 1
# ----------------------------------------------------------------------
prob = onp.load("vqsvd_problem_data.npz", allow_pickle=True)
S_true = prob["S_true"]
pauli_terms = [tuple(row) for row in prob["pauli_terms"]]
pauli_coeffs = prob["pauli_coeffs"]
n_qubits = int(prob["n_qubits"])
n_layers = int(prob["n_layers"])
T = int(prob["T"])
IDENTITY_SHIFT = float(prob["identity_shift"])
dim = 2 ** n_qubits

ckpt = onp.load("vqsvd_sym_checkpoint.npz")
theta = onp.array(ckpt["theta"])   # trained parameters, shape (n_layers, n_qubits, 2)
trained_iters = int(ckpt["iter"])

print("=" * 70)
print(f"Loaded trained ansatz ({trained_iters} training iterations), "
      f"{n_qubits} qubits, dim={dim}, extracting top-{T} singular values")
print("=" * 70)

# ----------------------------------------------------------------------
# 1. Noise levels (tunable)
# ----------------------------------------------------------------------
P1 = 0.0008   # single-qubit gate depolarizing rate (realistic superconducting-qubit scale)
P2 = 0.006    # two-qubit (CNOT) depolarizing rate, applied per touched qubit
P_READ = 0.015 # per-qubit readout bit-flip rate

dev = qml.device("default.mixed", wires=n_qubits)


def int_to_bits(i, n):
    return [int(b) for b in format(i, f"0{n}b")]


basis_patterns = [int_to_bits(i, n_qubits) for i in range(dim)]

# ----------------------------------------------------------------------
# 2. Forward / inverse ansatz layer, with noise channels interleaved
#    (p1 = p2 = 0 reproduces the exact noiseless circuit)
# ----------------------------------------------------------------------
def layer_forward(layer_params, p1, p2):
    for q in range(n_qubits):
        qml.RY(layer_params[q, 0], wires=q)
        if p1 > 0:
            qml.DepolarizingChannel(p1, wires=q)
        qml.RZ(layer_params[q, 1], wires=q)
        if p1 > 0:
            qml.DepolarizingChannel(p1, wires=q)
    for q in range(n_qubits):
        qml.CNOT(wires=[q, (q + 1) % n_qubits])
        if p2 > 0:
            qml.DepolarizingChannel(p2, wires=q)
            qml.DepolarizingChannel(p2, wires=(q + 1) % n_qubits)


def layer_inverse(layer_params, p1, p2):
    for q in reversed(range(n_qubits)):
        qml.CNOT(wires=[q, (q + 1) % n_qubits])
        if p2 > 0:
            qml.DepolarizingChannel(p2, wires=q)
            qml.DepolarizingChannel(p2, wires=(q + 1) % n_qubits)
    for q in range(n_qubits):
        qml.RZ(-layer_params[q, 1], wires=q)
        if p1 > 0:
            qml.DepolarizingChannel(p1, wires=q)
        qml.RY(-layer_params[q, 0], wires=q)
        if p1 > 0:
            qml.DepolarizingChannel(p1, wires=q)


def ansatz_forward(params, p1, p2):
    for l in range(n_layers):
        layer_forward(params[l], p1, p2)


def ansatz_inverse(params, p1, p2):
    for l in reversed(range(n_layers)):
        layer_inverse(params[l], p1, p2)


def folded_ansatz(params, scale_factor, p1, p2):
    """scale_factor: 1, 3, 5, ... -- unitary folding of the WHOLE ansatz."""
    ansatz_forward(params, p1, p2)
    extra_pairs = (scale_factor - 1) // 2
    for _ in range(extra_pairs):
        ansatz_inverse(params, p1, p2)
        ansatz_forward(params, p1, p2)


# ----------------------------------------------------------------------
# 3. Circuit that measures ONE Pauli term for ONE basis-state input
# ----------------------------------------------------------------------
@qml.qnode(dev)
def pauli_term_probs(params, basis_index, labels, scale_factor, p1, p2, p_read):
    qml.BasisState(onp.array(basis_index), wires=range(n_qubits))
    folded_ansatz(params, scale_factor, p1, p2)
    for q, lab in enumerate(labels):
        if lab == "X":
            qml.Hadamard(wires=q)
        # 'Z' and 'I' need no basis rotation
    if p_read > 0:
        for q in range(n_qubits):
            qml.BitFlip(p_read, wires=q)
    return qml.probs(wires=range(n_qubits))


def pauli_expectation_from_probs(probs_vec, labels, n_qubits):
    """Combine a probability distribution into a Pauli-string expectation."""
    support = [q for q, lab in enumerate(labels) if lab != "I"]
    if not support:
        return 1.0   # all-identity term
    total = 0.0
    for outcome in range(len(probs_vec)):
        bits = int_to_bits(outcome, n_qubits)
        sign = 1
        for q in support:
            if bits[q] == 1:
                sign *= -1
        total += sign * probs_vec[outcome]
    return total


def readout_damping_factor(labels, p_read):
    support_size = sum(1 for lab in labels if lab != "I")
    return (1 - 2 * p_read) ** support_size


# ----------------------------------------------------------------------
# 4. Estimate <psi_i|M|psi_i> under each condition
# ----------------------------------------------------------------------
def estimate_expectation(basis_index, scale_factor, p1, p2, p_read, apply_readout_correction):
    total = IDENTITY_SHIFT   # <psi|I|psi> = 1 always; this term needs no measurement
    for labels, coef in zip(pauli_terms, pauli_coeffs):
        probs_vec = pauli_term_probs(theta, basis_patterns[basis_index], labels,
                                      scale_factor, p1, p2, p_read)
        raw_term = pauli_expectation_from_probs(probs_vec, labels, n_qubits)
        if apply_readout_correction and p_read > 0:
            damp = readout_damping_factor(labels, p_read)
            raw_term = raw_term / damp
        total += coef * raw_term
    return total


print(f"\nNoise levels: single-qubit gate p1={P1}, CNOT p2={P2}, readout p_read={P_READ}\n")
t0 = time.time()

results = {name: [] for name in
           ["ideal", "noisy_raw", "readout_only", "zne_only", "combined"]}

for i in range(T):
    print(f"--- Singular vector index {i} ---")

    # 1. Ideal (no noise at all)
    ideal_val = estimate_expectation(i, 1, 0.0, 0.0, 0.0, apply_readout_correction=False)
    results["ideal"].append(ideal_val)

    # 2. Noisy raw (scale=1, gate + readout noise, no correction)
    noisy_raw_val = estimate_expectation(i, 1, P1, P2, P_READ, apply_readout_correction=False)
    results["noisy_raw"].append(noisy_raw_val)

    # 3. Readout mitigation only (scale=1, correct for readout damping)
    readout_val = estimate_expectation(i, 1, P1, P2, P_READ, apply_readout_correction=True)
    results["readout_only"].append(readout_val)

    # 4/5. ZNE (with and without readout correction) across scale factors
    scale_factors = [1, 3, 5]
    zne_raw_points = []
    zne_corrected_points = []
    for s in scale_factors:
        if s == 1:
            raw_pt, corr_pt = noisy_raw_val, readout_val
        else:
            raw_pt = estimate_expectation(i, s, P1, P2, P_READ, apply_readout_correction=False)
            corr_pt = estimate_expectation(i, s, P1, P2, P_READ, apply_readout_correction=True)
        zne_raw_points.append(raw_pt)
        zne_corrected_points.append(corr_pt)

    a1, b1 = onp.polyfit(scale_factors, zne_raw_points, 1)
    zne_only_val = b1
    a2, b2 = onp.polyfit(scale_factors, zne_corrected_points, 1)
    combined_val = b2

    results["zne_only"].append(zne_only_val)
    results["combined"].append(combined_val)

    print(f"  classical target       : {S_true[i]:.4f}")
    print(f"  ideal (no noise)       : {ideal_val:.4f}  (err {abs(ideal_val-S_true[i]):.4f})")
    print(f"  noisy, no mitigation   : {noisy_raw_val:.4f}  (err {abs(noisy_raw_val-S_true[i]):.4f})")
    print(f"  readout mitigation only: {readout_val:.4f}  (err {abs(readout_val-S_true[i]):.4f})")
    print(f"  ZNE only               : {zne_only_val:.4f}  (err {abs(zne_only_val-S_true[i]):.4f})")
    print(f"  ZNE + readout combined : {combined_val:.4f}  (err {abs(combined_val-S_true[i]):.4f})")
    print()

print(f"Total evaluation time: {time.time()-t0:.1f}s\n")

# ----------------------------------------------------------------------
# 5. Summary table across all T singular vectors
# ----------------------------------------------------------------------
print("=" * 70)
print("SUMMARY -- mean absolute error across top-{} singular values".format(T))
print("=" * 70)
for name in ["ideal", "noisy_raw", "readout_only", "zne_only", "combined"]:
    vals = onp.array(results[name])
    mean_err = onp.mean(onp.abs(vals - S_true[:T]))
    print(f"  {name:<15s}: mean abs error = {mean_err:.4f}")