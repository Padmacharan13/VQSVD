"""
error_mitigation_demo_visualized.py
Adds Matplotlib visualization layers to the ZNE and Readout mitigation demo.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


# 0. Setup

theta = np.pi / 3
shots = 20000
ideal_p0 = np.cos(theta / 2) ** 2
print(f"Exact ideal P(0) for theta = pi/3:  {ideal_p0:.4f}\n")

# Build a noise model
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

sim_ideal = AerSimulator()
sim_noisy = AerSimulator(noise_model=noise_model)

def p0_from_counts(counts):
    total = sum(counts.values())
    return counts.get("0", 0) / total

# 1. Baseline: ideal vs. plain noisy result (no mitigation)

def base_circuit():
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)
    return qc

ideal_counts = sim_ideal.run(base_circuit(), shots=shots).result().get_counts()
noisy_counts = sim_noisy.run(base_circuit(), shots=shots).result().get_counts()

p0_ideal_sim = p0_from_counts(ideal_counts)
p0_noisy_raw = p0_from_counts(noisy_counts)


# 2. Zero-Noise Extrapolation (ZNE)

def folded_circuit(scale_factor):
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    extra_pairs = (scale_factor - 1) // 2
    for _ in range(extra_pairs):
        qc.ry(-theta, 0)
        qc.ry(theta, 0)
    qc.measure(0, 0)
    return qc

scale_factors = [1, 3, 5]
zne_results = []
for s in scale_factors:
    counts = sim_noisy.run(folded_circuit(s), shots=shots).result().get_counts()
    zne_results.append(p0_from_counts(counts))

a, b = np.polyfit(scale_factors, zne_results, 1)
zne_extrapolated = b


# 3. Readout error mitigation

def calibration_counts(prepared_bit):
    qc = QuantumCircuit(1, 1)
    if prepared_bit == 1:
        qc.x(0)
    qc.measure(0, 0)
    return sim_noisy.run(qc, shots=shots).result().get_counts()

cal0 = calibration_counts(0)
cal1 = calibration_counts(1)

p0_given_0 = p0_from_counts(cal0)
p0_given_1 = p0_from_counts(cal1)

A = np.array([
    [p0_given_0,     p0_given_1],
    [1 - p0_given_0, 1 - p0_given_1]
])
A_inv = np.linalg.inv(A)

measured_vector = np.array([p0_noisy_raw, 1 - p0_noisy_raw])
corrected_vector = A_inv @ measured_vector
p0_readout_corrected = corrected_vector[0]


# 4. Combined ZNE + Readout Mitigation

def readout_correct(p0_raw):
    m_vec = np.array([p0_raw, 1 - p0_raw])
    corr = A_inv @ m_vec
    return corr[0]

combined_results = []
for s in scale_factors:
    counts = sim_noisy.run(folded_circuit(s), shots=shots).result().get_counts()
    raw_p0 = p0_from_counts(counts)
    combined_results.append(readout_correct(raw_p0))

a2, b2 = np.polyfit(scale_factors, combined_results, 1)
combined_extrapolated = b2


# ENHANCED VISUALIZATION BLOCK

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6))

# --- Plot 1: Zero-Noise Extrapolation Behavior ---
fit_scales = np.linspace(0, 5.5, 100)
fit_line_raw = a * fit_scales + b
fit_line_comb = a2 * fit_scales + b2

# Reference baseline line (Exact Ideal)
ax1.axhline(ideal_p0, color='forestgreen', linestyle=':', linewidth=2.5, label='Exact Analytical Ideal P(0) = 0.7500', zorder=1)

# Plot linear fit projections
ax1.plot(fit_scales, fit_line_raw, color='crimson', linestyle='--', alpha=0.8, linewidth=1.8, label='Linear ZNE Fit (Raw Data)')
ax1.plot(fit_scales, fit_line_comb, color='royalblue', linestyle='-', alpha=0.8, linewidth=1.8, label='Linear ZNE Fit (Readout-Mitigated)')

# Plot experimental noisy points vs combined points
ax1.scatter(scale_factors, zne_results, color='crimson', marker='o', s=90, zorder=4, label='Raw Noisy Data (r = 1, 3, 5)')
ax1.scatter(scale_factors, combined_results, color='royalblue', marker='s', s=90, zorder=4, label='Readout-Mitigated Data (r = 1, 3, 5)')

# Highlight extrapolated zero-noise intercept points at r = 0
ax1.scatter(0, zne_extrapolated, color='darkred', marker='X', s=160, zorder=5, label=f'ZNE Only Intercept (P(0)={zne_extrapolated:.4f})')
ax1.scatter(0, combined_extrapolated, color='navy', marker='X', s=160, zorder=5, label=f'ZNE + Readout Intercept (P(0)={combined_extrapolated:.4f})')

# Annotations explaining "Which is What" on ZNE Graph
ax1.annotate(r'$\mathbf{Zero\text{-}Noise\ Limit\ (r=0)}$' + '\n' + r'Stacked Mitigation $\approx$ Ideal Target', 
             xy=(0, combined_extrapolated), xytext=(0.55, 0.743),
             arrowprops=dict(arrowstyle="->", color="navy", lw=1.5),
             fontsize=9.5, fontweight='bold', color='navy',
             bbox=dict(boxstyle="round,pad=0.35", fc="aliceblue", ec="navy", lw=1.2))

ax1.annotate(r'$\mathbf{Base\ Noise\ Level\ (r=1)}$', 
             xy=(1, zne_results[0]), xytext=(1.4, 0.718),
             arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5),
             fontsize=9, fontweight='bold', color='darkred',
             bbox=dict(boxstyle="round,pad=0.3", fc="snow", ec="crimson", lw=1))

ax1.set_title("Zero-Noise Extrapolation (ZNE) Profile\n(Noise Amplification via Gate Folding & Zero-Limit Intercept)", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel("Noise Scale Factor ($r$)  [1 = Base Noise, 3/5 = Amplified Gate Noise]", fontsize=10.5)
ax1.set_ylabel("Probability of Outcome $P(0)$", fontsize=10.5)
ax1.set_xlim(-0.4, 5.4)
ax1.set_ylim(0.68, 0.755)
ax1.legend(frameon=True, loc='lower left', fontsize=8.5, framealpha=0.95)

# --- Plot 2: Readout Calibration Confusion Matrix Heatmap ---
ax2.grid(False)  # Turn off Seaborn background grid overlay on the matrix!
im = ax2.imshow(A, cmap='Blues', vmin=0, vmax=1, aspect='equal')

# Draw explicit gridlines separating the 4 matrix cells
ax2.set_xticks([0.5], minor=True)
ax2.set_yticks([0.5], minor=True)
ax2.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
ax2.tick_params(which='minor', size=0)

# Cell descriptive text explaining "Which is What" in Confusion Matrix
cell_labels = [
    [f"P(Meas 0 | Prep 0)\nTrue $|0\\rangle$ Retained\n{A[0,0]:.4f}", f"P(Meas 0 | Prep 1)\nReadout Error (1→0)\n{A[0,1]:.4f}"],
    [f"P(Meas 1 | Prep 0)\nReadout Error (0→1)\n{A[1,0]:.4f}", f"P(Meas 1 | Prep 1)\nTrue $|1\\rangle$ Retained\n{A[1,1]:.4f}"]
]

for i in range(2):
    for j in range(2):
        text_color = "white" if A[i, j] > 0.5 else "black"
        ax2.text(j, i, cell_labels[i][j], ha="center", va="center", color=text_color, fontweight='bold', fontsize=10)

ax2.set_title("Readout Calibration Confusion Matrix ($A$)\nMatrix Inversion: $P_{ideal} = A^{-1} \\cdot P_{measured}$", fontsize=12, fontweight='bold', pad=10)
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels([r'Prepared $|0\rangle$', r'Prepared $|1\rangle$'], fontsize=11, fontweight='bold')
ax2.set_yticklabels([r'Measured $0$', r'Measured $1$'], fontsize=11, fontweight='bold')
ax2.set_xlabel("True Prepared Quantum State ($j$)", fontsize=10.5, labelpad=8)
ax2.set_ylabel("Observed Measured Outcome ($i$)", fontsize=10.5, labelpad=8)

# Colorbar fixing: specify ax=ax2 explicitly so it doesn't overwrite ax2!
cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label("Transition Probability $P(i | j)$", fontsize=10)

plt.tight_layout()
plt.savefig("Figure_1.png", dpi=300, bbox_inches='tight')
plt.close()

# Print text summary to console
print("=" * 60)
print("SUMMARY RESULTS")
print("=" * 60)
print(f"Exact Analytical Target Value : {ideal_p0:.4f}")
print(f"Raw Distorted Noisy Result     : {p0_noisy_raw:.4f}  (Error: {abs(p0_noisy_raw - ideal_p0):.4f})")
print(f"ZNE Mitigated Only Result      : {zne_extrapolated:.4f}  (Error: {abs(zne_extrapolated - ideal_p0):.4f})")
print(f"Readout Mitigated Only Result  : {p0_readout_corrected:.4f}  (Error: {abs(p0_readout_corrected - ideal_p0):.4f})")
print(f"Stacked Combined (ZNE+Readout) : {combined_extrapolated:.4f}  (Error: {abs(combined_extrapolated - ideal_p0):.4f})")


# ----------------------------------------------------------------------
# FIGURE 2 -- Presentation-friendly summary: all four methods side by side
# ----------------------------------------------------------------------
methods = ["Noisy\n(no mitigation)", "ZNE\nonly", "Readout\nonly", "ZNE + Readout\n(combined)"]
values = [p0_noisy_raw, zne_extrapolated, p0_readout_corrected, combined_extrapolated]
errors = [abs(v - ideal_p0) for v in values]
bar_colors = ["crimson", "darkorange", "royalblue", "seagreen"]

fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Left panel: estimated P(0) for each method, vs. the true ideal line ---
bars = bx1.bar(methods, values, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.8)
bx1.axhline(ideal_p0, color="forestgreen", linestyle=":", linewidth=2.5,
            label=f"Exact ideal P(0) = {ideal_p0:.4f}")
for bar, val in zip(bars, values):
    bx1.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.4f}",
              ha="center", fontsize=9.5, fontweight="bold")
bx1.set_ylim(min(values) - 0.02, ideal_p0 + 0.02)
bx1.set_title("Estimated P(0) by Mitigation Method", fontsize=12, fontweight="bold")
bx1.set_ylabel("Probability of outcome P(0)")
bx1.legend(loc="lower right", fontsize=9)

# --- Right panel: absolute error for each method (lower = better) ---
bars2 = bx2.bar(methods, errors, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.8)
for bar, err in zip(bars2, errors):
    bx2.text(bar.get_x() + bar.get_width() / 2, err + 0.0008, f"{err:.4f}",
              ha="center", fontsize=9.5, fontweight="bold")
bx2.set_title("Absolute Error vs. Ideal Value\n(lower is better)", fontsize=12, fontweight="bold")
bx2.set_ylabel("Absolute error |estimate - ideal|")

plt.tight_layout()
plt.savefig("Figure_2.png", dpi=300, bbox_inches="tight")
plt.close()