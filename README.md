# Error Mitigation Demo (ZNE + Readout)

This repository contains a small Qiskit demo that demonstrates two quantum error mitigation techniques:
- Zero-Noise Extrapolation (ZNE)
- Readout (measurement) error mitigation

The main script visualizes how these techniques improve the estimated probability P(0) for a single-qubit RY rotation.

Files
- `error_mitigation_test.py`: Demo script that builds a noisy Aer simulator, runs baseline, ZNE, readout calibration, and combined mitigation; shows Matplotlib plots and prints a numeric summary.
- `requirements.txt`: Python dependencies used by the project.

Requirements
- Python 3.8+ (3.10+ recommended)
- Install dependencies:

```bash
python -m pip install -r requirements.txt
```

What the script does
1. Defines a single-qubit RY rotation by `theta = pi/3` and computes the exact analytical probability P(0).
2. Builds a simple NoiseModel with depolarizing gate error and a readout confusion matrix.
3. Runs three experiments:
   - Baseline (ideal vs noisy)
   - ZNE: runs folded circuits with noise scale factors and linearly extrapolates to zero noise
   - Readout mitigation: measures calibration circuits to build a confusion matrix and inverts it
4. Also runs the combined flow: readout-corrected probabilities for each folded circuit and a ZNE extrapolation on those corrected values.
5. Visualizes results with Matplotlib: a ZNE profile plot and a readout confusion-matrix heatmap, then prints a short numeric summary.

Usage
```bash
python error_mitigation_test.py
```

Expected output
- A Matplotlib window with two subplots:
  - Left: noisy vs readout-corrected points and the ZNE extrapolated intercepts at scale=0, plus the analytical ideal line.
  - Right: the 2x2 readout confusion matrix (heatmap) with numeric annotations.
- A console summary with the exact ideal value, raw noisy result, ZNE-only result, readout-corrected result, and combined ZNE+Readout result (with errors).

Customization
- Change `theta` and `shots` at the top of `error_mitigation_test.py`.
- Adjust `gate_error_rate`, `readout_p0_to_1`, and `readout_p1_to_0` to simulate different noise strengths.

Notes & Troubleshooting
- If `qiskit-aer` fails to import, ensure you installed the correct Aer version for your Python interpreter. Installing the Qiskit metapackage (`pip install qiskit`) can also resolve dependency mismatches.
- On some platforms, `qiskit-aer` may require a precompiled wheel; use the Python version recommended in the package docs.

License
- No license specified. Use and modify freely or add a license file if needed.
