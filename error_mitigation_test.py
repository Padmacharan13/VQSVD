"""
error_mitigation_demo.py

A small, self-contained demo showing TWO error-mitigation techniques
used in the EM-VQSVD project:

    1. Zero-Noise Extrapolation (ZNE)
    2. Readout error mitigation

We use a single, simple circuit (one qubit, one RY rotation) instead of
the full VQSVD circuit so the effect of noise and mitigation is easy to
see and verify against a known, exact answer.

Circuit:  RY(theta) on |0>, then measure in the Z basis.
Ideal probability of measuring '0'  =  cos^2(theta / 2)   (exact formula)

Requires: qiskit, qiskit-aer   (pip install qiskit qiskit-aer)
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

# ----------------------------------------------------------------------
# 0. Setup
# ----------------------------------------------------------------------
theta = np.pi / 3          # our test angle
shots = 20000
ideal_p0 = np.cos(theta / 2) ** 2
print(f"Exact ideal P(0) for theta = pi/3:  {ideal_p0:.4f}\n")

# Build a noise model: every RY gate has a small depolarizing error,
# and every measurement has a readout (bit-flip) error.
gate_error_rate = 0.03      # 3% depolarizing error per RY gate
readout_p0_to_1 = 0.05      # 5% chance a true 0 is read as 1
readout_p1_to_0 = 0.03      # 3% chance a true 1 is read as 0

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(
    depolarizing_error(gate_error_rate, 1), ["ry"]
)
noise_model.add_all_qubit_readout_error(
    ReadoutError([[1 - readout_p0_to_1, readout_p0_to_1],
                  [readout_p1_to_0, 1 - readout_p1_to_0]])
)

sim_ideal = AerSimulator()                                   # no noise
sim_noisy = AerSimulator(noise_model=noise_model)             # with noise


def p0_from_counts(counts):
    """Return P(measured '0') from a Qiskit counts dictionary."""
    total = sum(counts.values())
    return counts.get("0", 0) / total


# ----------------------------------------------------------------------
# 1. Baseline: ideal vs. plain noisy result (no mitigation)
# ----------------------------------------------------------------------
def base_circuit():
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)
    return qc


ideal_counts = sim_ideal.run(base_circuit(), shots=shots).result().get_counts()
noisy_counts = sim_noisy.run(base_circuit(), shots=shots).result().get_counts()

p0_ideal_sim = p0_from_counts(ideal_counts)
p0_noisy_raw = p0_from_counts(noisy_counts)

print("Step 1: baseline comparison")
print(f"  Ideal simulator P(0)      : {p0_ideal_sim:.4f}")
print(f"  Noisy simulator P(0) raw  : {p0_noisy_raw:.4f}  <-- noise pulls this away from {ideal_p0:.4f}\n")

# ----------------------------------------------------------------------
# 2. Zero-Noise Extrapolation (ZNE)
# ----------------------------------------------------------------------
# Idea: amplify the noise on purpose by "gate folding" -- replace RY(theta)
# with RY(theta) -> RY(-theta) -> RY(theta) -> RY(-theta) -> RY(theta) ...
# Mathematically this is still the identity operation composed with the
# original gate, so the IDEAL result never changes -- but each extra pair
# of gates adds more real noise on hardware/noisy-sim. This lets us sample
# the result at several noise levels (scale factors 1, 3, 5, ...) and
# extrapolate back to the hypothetical "zero noise" point.

def folded_circuit(scale_factor):
    """scale_factor must be an odd integer: 1, 3, 5, ..."""
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    extra_pairs = (scale_factor - 1) // 2
    for _ in range(extra_pairs):
        qc.ry(-theta, 0)   # undo
        qc.ry(theta, 0)    # redo  -> net effect = identity, but 2 more noisy gates
    qc.measure(0, 0)
    return qc


scale_factors = [1, 3, 5]
zne_results = []
for s in scale_factors:
    counts = sim_noisy.run(folded_circuit(s), shots=shots).result().get_counts()
    zne_results.append(p0_from_counts(counts))

# Linear fit: p0(scale) = a * scale + b, then extrapolate to scale = 0
a, b = np.polyfit(scale_factors, zne_results, 1)
zne_extrapolated = b   # value at scale_factor = 0

print("Step 2: Zero-Noise Extrapolation (ZNE)")
for s, r in zip(scale_factors, zne_results):
    print(f"  Noise scale {s}x -> P(0) = {r:.4f}")
print(f"  ZNE extrapolated (scale=0) P(0) : {zne_extrapolated:.4f}")
print(f"  Compare to exact ideal          : {ideal_p0:.4f}")
print(f"  Error before ZNE : {abs(p0_noisy_raw - ideal_p0):.4f}")
print(f"  Error after  ZNE : {abs(zne_extrapolated - ideal_p0):.4f}\n")

# ----------------------------------------------------------------------
# 3. Readout error mitigation
# ----------------------------------------------------------------------
# Idea: calibrate how often a KNOWN state gets misread, build a small
# "confusion matrix", then invert it and apply it to correct the raw
# noisy counts from our actual circuit.

def calibration_counts(prepared_bit):
    qc = QuantumCircuit(1, 1)
    if prepared_bit == 1:
        qc.x(0)               # prepare |1>  (X gate itself is treated as noiseless here
    qc.measure(0, 0)           # so we isolate pure readout error for this demo)
    return sim_noisy.run(qc, shots=shots).result().get_counts()


cal0 = calibration_counts(0)   # prepared |0>, see what we measure
cal1 = calibration_counts(1)   # prepared |1>, see what we measure

p0_given_0 = p0_from_counts(cal0)          # should be ~1 if perfect
p0_given_1 = p0_from_counts(cal1)          # should be ~0 if perfect

# Confusion matrix A: columns = true state, rows = measured state
#   A @ [true P(0), true P(1)]^T  =  [measured P(0), measured P(1)]^T
A = np.array([
    [p0_given_0,     p0_given_1],
    [1 - p0_given_0, 1 - p0_given_1]
])
A_inv = np.linalg.inv(A)

print("Step 3: Readout error mitigation")
print(f"  Calibration: measured P(0) when true state is |0> : {p0_given_0:.4f}")
print(f"  Calibration: measured P(0) when true state is |1> : {p0_given_1:.4f}")

# Apply correction to our ORIGINAL noisy circuit result (scale=1, from Step 1)
measured_vector = np.array([p0_noisy_raw, 1 - p0_noisy_raw])
corrected_vector = A_inv @ measured_vector
p0_readout_corrected = corrected_vector[0]

print(f"  Raw noisy P(0)                 : {p0_noisy_raw:.4f}")
print(f"  Readout-corrected P(0)         : {p0_readout_corrected:.4f}")
print(f"  Compare to exact ideal         : {ideal_p0:.4f}")
print(f"  Error before mitigation        : {abs(p0_noisy_raw - ideal_p0):.4f}")
print(f"  Error after readout mitigation : {abs(p0_readout_corrected - ideal_p0):.4f}\n")

# ----------------------------------------------------------------------
# 4. Combine BOTH: apply readout mitigation at every noise scale factor,
#    THEN run ZNE extrapolation on the already-corrected values.
# ----------------------------------------------------------------------
def readout_correct(p0_raw):
    """Apply the same confusion-matrix inversion to any raw P(0) value."""
    measured_vector = np.array([p0_raw, 1 - p0_raw])
    corrected = A_inv @ measured_vector
    return corrected[0]


combined_results = []
for s in scale_factors:
    counts = sim_noisy.run(folded_circuit(s), shots=shots).result().get_counts()
    raw_p0 = p0_from_counts(counts)
    corrected_p0 = readout_correct(raw_p0)   # remove readout bias first
    combined_results.append(corrected_p0)

a2, b2 = np.polyfit(scale_factors, combined_results, 1)
combined_extrapolated = b2   # ZNE extrapolation of the readout-corrected values

print("Step 4: Combined ZNE + readout mitigation")
for s, r in zip(scale_factors, combined_results):
    print(f"  Noise scale {s}x -> readout-corrected P(0) = {r:.4f}")
print(f"  Combined extrapolated (scale=0) P(0) : {combined_extrapolated:.4f}")
print(f"  Error after combined mitigation       : {abs(combined_extrapolated - ideal_p0):.4f}\n")

# ----------------------------------------------------------------------
# 5. Summary
# ----------------------------------------------------------------------
print("=" * 60)
print("SUMMARY  (closer to the ideal value is better)")
print("=" * 60)
print(f"  Exact ideal value              : {ideal_p0:.4f}")
print(f"  Noisy, no mitigation           : {p0_noisy_raw:.4f}  (error {abs(p0_noisy_raw - ideal_p0):.4f})")
print(f"  Noisy + ZNE only               : {zne_extrapolated:.4f}  (error {abs(zne_extrapolated - ideal_p0):.4f})")
print(f"  Noisy + readout mitigation only: {p0_readout_corrected:.4f}  (error {abs(p0_readout_corrected - ideal_p0):.4f})")
print(f"  Noisy + ZNE + readout combined : {combined_extrapolated:.4f}  (error {abs(combined_extrapolated - ideal_p0):.4f})")