"""
vqsvd_8x8.py
======================
VQSVD for a single 8x8 matrix (3 qubits) -- WITH noise + mitigation.

Pipeline:
  1. Build a real symmetric positive-definite test matrix (random-field Ising
     type: local X, Z and neighbour ZZ Pauli terms) -- this also gives an
     EXACT, cheap Pauli decomposition of M for free (needed in step 4).
  2. Classical ground truth (eigendecomposition).
  3. Train a VQSVD ansatz (RY rotations + CNOT ring) with the Ky Fan loss on
     the IDEAL simulator (several random restarts, best kept).
  4. Evaluate the TRAINED circuit for real, under noise, by measuring
     <psi|M|psi> via its Pauli decomposition (rotate-then-measure), under:
       - ideal simulator (sanity check, matches step 3)
       - noisy simulator, no mitigation (gate + readout noise)
       - readout error mitigation only (analytic damping-factor correction)
       - Zero-Noise Extrapolation (ZNE) only (unitary folding + extrapolation)
       - ZNE + readout combined
  5. Save a results table and graphs covering all of this.

Install:  pip install pennylane matplotlib numpy
Run:      python vqsvd_8x8.py
          python vqsvd_8x8.py --quick            # fast smoke test
          python vqsvd_8x8.py --restarts 5       # more training restarts
          python vqsvd_8x8.py --skip-noise       # training + ideal only
          python vqsvd_8x8.py --scales 1 3       # cheaper ZNE (drop 5x)
Training is checkpointed in params_8x8.npy -- re-run to resume/improve.

Produces:
  vqsvd_8x8_table.csv        -- loss/error summary + per-sigma detail
  vqsvd_8x8_mitigation.csv   -- ideal/noisy/readout/ZNE/combined, per sigma
  vqsvd_8x8_graphs.png       -- convergence + predicted-vs-actual
  vqsvd_8x8_mitigation.png   -- error by mitigation method
"""
import argparse
import csv
import os
import time

import numpy as onp
import pennylane as qml
from pennylane import numpy as np

# ----------------------------------------------------------------------
# Configuration for this matrix size
# ----------------------------------------------------------------------
N_QUBITS = 3
DIM = 2 ** N_QUBITS
T = 3                  # number of singular values/vectors to extract
LAYERS = 6        # ansatz depth
ITERS = 300          # training iterations per restart
RESTARTS = 2    # random restarts (best kept)
SEED = 11
TAG = "8x8"

INIT_SCALES = [0.1, 0.3, 0.5, 0.2, 0.4]

# Noise levels for the mitigation study (realistic current-hardware scale)
P1 = 0.0008     # single-qubit gate depolarizing rate
P2 = 0.006      # two-qubit (CNOT) depolarizing rate, applied per touched qubit
P_READ = 0.015  # per-qubit readout bit-flip rate


# ----------------------------------------------------------------------
# 0. Build the test matrix AND its Pauli-term decomposition
# ----------------------------------------------------------------------
def build_matrix(n):
    """Returns (M, terms, identity_shift). terms is a list of
    (coefficient, pauli_type, qubit_list) with pauli_type in {'X','Z','ZZ'}."""
    X = onp.array([[0, 1], [1, 0]])
    Z = onp.diag([1, -1])
    I2 = onp.eye(2)

    def op_at(op, q):
        mats = [I2] * n
        mats[q] = op
        m = mats[0]
        for a in mats[1:]:
            m = onp.kron(m, a)
        return m

    rng = onp.random.default_rng(SEED + n)
    M = onp.zeros((2 ** n, 2 ** n))
    terms = []
    for q in range(n):
        c = rng.uniform(0.5, 1.5)
        M += c * op_at(X, q); terms.append((c, "X", [q]))
        c = rng.uniform(0.2, 0.8)
        M += c * op_at(Z, q); terms.append((c, "Z", [q]))
    for q in range(n if n > 2 else 1):
        c = rng.uniform(0.5, 1.5)
        M += c * (op_at(Z, q) @ op_at(Z, (q + 1) % n))
        terms.append((c, "ZZ", [q, (q + 1) % n]))
    shift = abs(onp.linalg.eigvalsh(M).min()) + 0.5
    M += shift * onp.eye(2 ** n)
    return M, terms, shift


M_np, PAULI_TERMS, IDENTITY_SHIFT = build_matrix(N_QUBITS)
S_true = onp.sort(onp.linalg.eigvalsh(M_np))[::-1][:T]
M = np.array(M_np)
LOSS_OPT = float(sum((T - i) * S_true[i] for i in range(T)))

print("=" * 65)
print(f"Matrix: {DIM}x{DIM} ({N_QUBITS} qubits)  |  extracting top-{T} singular values")
print(f"Pauli terms for noisy measurement: {len(PAULI_TERMS)} + identity shift {IDENTITY_SHIFT:.3f}")
print("=" * 65)
print("Classical (ground truth) singular values:", onp.round(S_true, 4))
print(f"Optimal Ky Fan loss: {LOSS_OPT:.4f}\n")

# ----------------------------------------------------------------------
# 1. Ansatz + Ky Fan loss (ideal simulator, exact statevectors, for TRAINING)
# ----------------------------------------------------------------------
dev_ideal = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev_ideal, interface="autograd", diff_method="backprop")
def state(params, bits):
    qml.BasisState(np.array(bits), wires=range(N_QUBITS))
    for l in range(LAYERS):
        for q in range(N_QUBITS):
            qml.RY(params[l, q], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    for q in range(N_QUBITS):
        qml.RY(params[LAYERS, q], wires=q)
    return qml.state()


bits = [[int(b) for b in format(i, f"0{N_QUBITS}b")] for i in range(T)]
q_w = [float(T - i) for i in range(T)]


def ky_fan(params):
    tot = 0.0
    for i in range(T):
        u = state(params, bits[i])
        tot = tot + q_w[i] * np.real(np.sum(np.conj(u) * (M @ u)))
    return tot


def cost(p):
    return -ky_fan(p)


# ----------------------------------------------------------------------
# 2. Train (several restarts, resumable, best kept)
# ----------------------------------------------------------------------
def train(restarts, iters):
    ckpt = f"params_{TAG}.npy"
    best = None
    t0 = time.time()
    for r in range(restarts):
        rng = onp.random.default_rng(SEED * 1000 + N_QUBITS * 10 + r)
        if r == 0 and os.path.exists(ckpt):
            p0 = onp.load(ckpt)
            print(f"  restart 0: resumed from checkpoint {ckpt}")
        else:
            p0 = rng.uniform(-onp.pi, onp.pi, (LAYERS + 1, N_QUBITS)) * INIT_SCALES[r % len(INIT_SCALES)]
        p = np.array(p0, requires_grad=True)
        hist = []
        lr0, lr1 = 0.1, 0.01
        opt = qml.AdamOptimizer(lr0)
        for it in range(iters):
            opt.stepsize = lr0 * (lr1 / lr0) ** (it / iters)
            p, c = opt.step_and_cost(cost, p)
            hist.append(float(-c))
        final = float(ky_fan(p))
        print(f"  restart {r}: final loss {final:.4f} / optimum {LOSS_OPT:.4f}"
              f"  ({time.time()-t0:.0f}s)")
        if best is None or final > best[0]:
            best = (final, onp.array(p), hist)
    onp.save(ckpt, best[1])
    return best


# ----------------------------------------------------------------------
# 3. Noisy evaluation of the TRAINED circuit, via Pauli-term measurement
# ----------------------------------------------------------------------
dev_mixed = qml.device("default.mixed", wires=N_QUBITS)


def layer_forward(layer_params, p1, p2):
    for q in range(N_QUBITS):
        qml.RY(layer_params[q], wires=q)
        if p1 > 0:
            qml.DepolarizingChannel(p1, wires=q)
    for q in range(N_QUBITS):
        qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
        if p2 > 0:
            qml.DepolarizingChannel(p2, wires=q)
            qml.DepolarizingChannel(p2, wires=(q + 1) % N_QUBITS)


def layer_inverse(layer_params, p1, p2):
    for q in reversed(range(N_QUBITS)):
        qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
        if p2 > 0:
            qml.DepolarizingChannel(p2, wires=q)
            qml.DepolarizingChannel(p2, wires=(q + 1) % N_QUBITS)
    for q in range(N_QUBITS):
        qml.RY(-layer_params[q], wires=q)
        if p1 > 0:
            qml.DepolarizingChannel(p1, wires=q)


def ansatz_forward(params, p1, p2):
    for l in range(LAYERS):
        layer_forward(params[l], p1, p2)
    for q in range(N_QUBITS):
        qml.RY(params[LAYERS, q], wires=q)
        if p1 > 0:
            qml.DepolarizingChannel(p1, wires=q)


def ansatz_inverse(params, p1, p2):
    for q in range(N_QUBITS):
        qml.RY(-params[LAYERS, q], wires=q)
        if p1 > 0:
            qml.DepolarizingChannel(p1, wires=q)
    for l in reversed(range(LAYERS)):
        layer_inverse(params[l], p1, p2)


def folded_ansatz(params, scale_factor, p1, p2):
    ansatz_forward(params, p1, p2)
    for _ in range((scale_factor - 1) // 2):
        ansatz_inverse(params, p1, p2)
        ansatz_forward(params, p1, p2)


@qml.qnode(dev_mixed)
def pauli_term_probs(params, basis_idx_bits, pauli_type, qubits, scale_factor, p1, p2, p_read):
    qml.BasisState(onp.array(basis_idx_bits), wires=range(N_QUBITS))
    folded_ansatz(params, scale_factor, p1, p2)
    if pauli_type == "X":
        qml.Hadamard(wires=qubits[0])
    # 'Z' and 'ZZ' need no basis rotation
    if p_read > 0:
        for q in range(N_QUBITS):
            qml.BitFlip(p_read, wires=q)
    return qml.probs(wires=range(N_QUBITS))


def pauli_expectation(probs_vec, qubits):
    total = 0.0
    for outcome in range(len(probs_vec)):
        bitstr = format(outcome, f"0{N_QUBITS}b")
        sign = 1
        for q in qubits:
            if bitstr[q] == "1":
                sign *= -1
        total += sign * probs_vec[outcome]
    return total


def readout_damping(qubits, p_read):
    return (1 - 2 * p_read) ** len(qubits)


def estimate_expectation(params, basis_idx_bits, scale_factor, p1, p2, p_read, correct_readout):
    total = IDENTITY_SHIFT
    for coef, ptype, qubits in PAULI_TERMS:
        probs_vec = pauli_term_probs(params, basis_idx_bits, ptype, qubits,
                                      scale_factor, p1, p2, p_read)
        raw = pauli_expectation(probs_vec, qubits)
        if correct_readout and p_read > 0:
            raw = raw / readout_damping(qubits, p_read)
        total += coef * raw
    return total


def run_mitigation_study(params, scale_factors):
    results = {k: [] for k in ["ideal", "noisy_raw", "readout_only", "zne_only", "combined"]}
    for i in range(T):
        ideal_val = estimate_expectation(params, bits[i], 1, 0.0, 0.0, 0.0, False)
        raw_pts, corr_pts = [], []
        for s in scale_factors:
            raw_pts.append(estimate_expectation(params, bits[i], s, P1, P2, P_READ, False))
            corr_pts.append(estimate_expectation(params, bits[i], s, P1, P2, P_READ, True))
        noisy_raw_val = raw_pts[0]
        readout_val = corr_pts[0]
        if len(scale_factors) > 1:
            _, zne_only_val = onp.polyfit(scale_factors, raw_pts, 1)
            _, combined_val = onp.polyfit(scale_factors, corr_pts, 1)
        else:
            zne_only_val, combined_val = noisy_raw_val, readout_val
        results["ideal"].append(ideal_val)
        results["noisy_raw"].append(noisy_raw_val)
        results["readout_only"].append(readout_val)
        results["zne_only"].append(zne_only_val)
        results["combined"].append(combined_val)
        print(f"  sigma[{i}]  true={S_true[i]:.4f}  ideal={ideal_val:.4f}  "
              f"noisy={noisy_raw_val:.4f}  readout={readout_val:.4f}  "
              f"zne={zne_only_val:.4f}  combined={combined_val:.4f}")
    return results


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--restarts", type=int, default=RESTARTS)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--skip-noise", action="store_true",
                     help="skip the noise/mitigation study (training + ideal only)")
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 3, 5],
                     help="ZNE noise scale factors, e.g. --scales 1 3")
    args = ap.parse_args()
    iters = max(20, int(args.iters * (0.15 if args.quick else 1.0)))
    restarts = 1 if args.quick else args.restarts

    print(f"Training: {restarts} restart(s) x {iters} iterations\n")
    final_loss, p_best, hist = train(restarts, iters)

    sig = []
    for i in range(T):
        u = state(np.array(p_best), bits[i])
        sig.append(float(np.real(np.sum(np.conj(u) * (M @ u)))))
    sig = onp.array(sig)
    abs_err = onp.abs(sig - S_true)
    rel_err = abs_err / S_true

    print("\n" + "=" * 65)
    print(f"RESULTS -- {DIM}x{DIM} matrix ({N_QUBITS} qubits), ideal simulator")
    print("=" * 65)
    print(f"{'Index':<8}{'Classical sigma':<18}{'VQSVD sigma':<16}{'Abs err':<12}{'Rel err (%)':<10}")
    for i in range(T):
        print(f"{i:<8}{S_true[i]:<18.4f}{sig[i]:<16.4f}{abs_err[i]:<12.4f}{rel_err[i]*100:<10.3f}")
    print(f"\nOptimal loss   : {LOSS_OPT:.4f}")
    print(f"Final loss     : {final_loss:.4f}")
    print(f"Loss gap       : {LOSS_OPT - final_loss:.4f}")
    print(f"Mean abs error : {abs_err.mean():.4f}")
    print(f"Mean rel error : {rel_err.mean()*100:.3f}%")

    with open(f"vqsvd_{TAG}_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Matrix", "Qubits", "Layers", "Params", "Optimal loss",
                    "Final loss", "Loss gap", "Mean abs err", "Mean rel err (%)"])
        w.writerow([f"{DIM}x{DIM}", N_QUBITS, LAYERS, (LAYERS + 1) * N_QUBITS,
                    f"{LOSS_OPT:.4f}", f"{final_loss:.4f}", f"{LOSS_OPT-final_loss:.4f}",
                    f"{abs_err.mean():.4f}", f"{rel_err.mean()*100:.3f}"])
        w.writerow([])
        w.writerow(["Index", "Classical sigma", "VQSVD sigma", "Abs error", "Rel error (%)"])
        for i in range(T):
            w.writerow([i, f"{S_true[i]:.4f}", f"{sig[i]:.4f}",
                        f"{abs_err[i]:.4f}", f"{rel_err[i]*100:.3f}"])
    print(f"\nSaved table -> vqsvd_{TAG}_table.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    h = onp.array(hist) / LOSS_OPT
    ax1.plot(h, color="#3b6fb6")
    ax1.axhline(1.0, color="k", ls="--", lw=1, label="optimum")
    ax1.set_xlabel("Training iteration"); ax1.set_ylabel("Ky Fan loss / optimal loss")
    ax1.set_title(f"Convergence -- {DIM}x{DIM} matrix")
    ax1.legend(); ax1.grid(alpha=0.3)

    idx = onp.arange(T)
    w_ = 0.35
    ax2.bar(idx - w_/2, S_true, w_, label="Classical (actual)", color="#2e8b57")
    ax2.bar(idx + w_/2, sig, w_, label="VQSVD (predicted)", color="#3b6fb6")
    ax2.set_xticks(idx); ax2.set_xlabel("Singular value index"); ax2.set_ylabel("Singular value")
    ax2.set_title(f"Predicted vs. actual -- {DIM}x{DIM} matrix")
    ax2.legend()
    for i in range(T):
        ax2.text(i, max(S_true[i], sig[i]) * 1.01, f"{rel_err[i]*100:.2f}%", ha="center", fontsize=8)
    fig.suptitle(f"VQSVD on a {DIM}x{DIM} matrix ({N_QUBITS} qubits)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"vqsvd_{TAG}_graphs.png", dpi=200)
    print(f"Saved graphs -> vqsvd_{TAG}_graphs.png")

    if args.skip_noise:
        print("\n--skip-noise set: skipping the noise/mitigation study.")
        return

    print("\n" + "=" * 65)
    print(f"NOISE + MITIGATION STUDY  (p1={P1}, p2={P2}, p_read={P_READ}, "
          f"scales={args.scales})")
    print(f"({len(PAULI_TERMS)} Pauli terms x {T} vectors x {len(args.scales)} scales "
          f"= {len(PAULI_TERMS)*T*len(args.scales)} circuit evaluations -- may take a while "
          f"for larger matrices)")
    print("=" * 65)
    t0 = time.time()
    mit = run_mitigation_study(onp.array(p_best), args.scales)
    print(f"Mitigation study took {time.time()-t0:.0f}s\n")

    with open(f"vqsvd_{TAG}_mitigation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Index", "Classical", "Ideal", "Noisy (no mitigation)",
                    "Readout only", "ZNE only", "ZNE+Readout combined"])
        for i in range(T):
            w.writerow([i, f"{S_true[i]:.4f}", f"{mit['ideal'][i]:.4f}",
                        f"{mit['noisy_raw'][i]:.4f}", f"{mit['readout_only'][i]:.4f}",
                        f"{mit['zne_only'][i]:.4f}", f"{mit['combined'][i]:.4f}"])
        w.writerow([])
        w.writerow(["Method", "Mean abs error"])
        for k in ["ideal", "noisy_raw", "readout_only", "zne_only", "combined"]:
            vals = onp.array(mit[k])
            w.writerow([k, f"{onp.mean(onp.abs(vals - S_true)):.4f}"])
    print(f"Saved mitigation table -> vqsvd_{TAG}_mitigation.csv")

    print("\nSUMMARY -- mean absolute error across top-{} singular values:".format(T))
    mean_errs = {}
    for k in ["ideal", "noisy_raw", "readout_only", "zne_only", "combined"]:
        vals = onp.array(mit[k])
        mean_errs[k] = float(onp.mean(onp.abs(vals - S_true)))
        print(f"  {k:<15s}: {mean_errs[k]:.4f}")

    fig2, ax = plt.subplots(figsize=(8, 5))
    labels = ["Ideal", "Noisy\n(no mitigation)", "Readout\nonly", "ZNE\nonly",
              "ZNE+Readout\n(combined)"]
    keys = ["ideal", "noisy_raw", "readout_only", "zne_only", "combined"]
    vals = [mean_errs[k] for k in keys]
    colors = ["#2e8b57", "#c0392b", "#3b6fb6", "#e67e22", "#8e44ad"]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v, f"{v:.4f}", ha="center",
                va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean absolute error")
    ax.set_title(f"Error by mitigation method -- {DIM}x{DIM} matrix ({N_QUBITS} qubits)")
    fig2.tight_layout()
    fig2.savefig(f"vqsvd_{TAG}_mitigation.png", dpi=200)
    print(f"Saved mitigation graph -> vqsvd_{TAG}_mitigation.png")


if __name__ == "__main__":
    main()