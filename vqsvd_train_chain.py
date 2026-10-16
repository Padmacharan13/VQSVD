"""
vqsvd_train_chain.py   (v2, fast)
=================================
Train the HARDWARE-FRIENDLY VQSVD ansatz on the ideal simulator, for any size
from 4x4 up to 1024x1024.

Ansatz: RY rotations + a LINEAR CHAIN of CNOTs (0-1, 1-2, ..., n-2 - n-1), no
ring wrap-around, so it maps onto a path of qubits on IBM chips with no SWAPs.
Loss: Ky Fan loss  L = sum_i q_i <i|U^T M U|i>,  q = (T, ..., 1)  (maximised).

Speed: M is a short sum of Pauli terms, so <i|U^T M U|i> is evaluated as a
Pauli-sum expectation on PennyLane's fast C++ simulator (lightning.qubit) with
adjoint differentiation. This is ~10x faster than dense matrix-vector
products and keeps training practical up to 10 qubits (1024x1024).

    pip install pennylane pennylane-lightning
    python vqsvd_train_chain.py --size 512x512
    python vqsvd_train_chain.py --size 1024x1024 --layers 36 --iters 1500
    python vqsvd_train_chain.py --size 1024x1024 --resume --iters 600     # keep improving

Writes params_<size>_chain.npy (used by vqsvd_ibm_hardware.py --entangler chain).
Needs vqsvd_ibm_hardware.py in the same folder (shared matrix definition).
"""
import argparse
import os
import time

import numpy as onp
import pennylane as qml
from pennylane import numpy as np

from vqsvd_ibm_hardware import SIZES, build_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--size", required=True, choices=list(SIZES))
ap.add_argument("--iters", type=int, default=800, help="iterations per run")
ap.add_argument("--restarts", type=int, default=2, help="random restarts (fresh run only)")
ap.add_argument("--layers", type=int, default=None, help="ansatz depth (default from SIZES)")
ap.add_argument("--resume", action="store_true", help="warm-start from the saved parameters")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="lightning.qubit",
                help="PennyLane device (falls back to default.qubit if unavailable)")
args = ap.parse_args()

cfg = SIZES[args.size]
n, T = cfg["n"], cfg["T"]
L = args.layers or cfg["layers"]
ckpt = f"params_{args.size}_chain.npy"

t_build = time.time()
M_np, terms, shift = build_matrix(n)
S_true = onp.sort(onp.linalg.eigvalsh(M_np))[::-1][:T]
t0c = time.perf_counter()
onp.linalg.svd(M_np)
classical_time = time.perf_counter() - t0c
q_w = [float(T - i) for i in range(T)]
LOSS_OPT = float(sum(q_w[i] * S_true[i] for i in range(T)))


def pauli_sum():
    ops, cs = [], []
    for c, pt, qs in terms:
        if pt == "X":
            op = qml.PauliX(qs[0])
        elif pt == "Z":
            op = qml.PauliZ(qs[0])
        else:
            op = qml.PauliZ(qs[0]) @ qml.PauliZ(qs[1])
        ops.append(op)
        cs.append(c)
    return qml.dot(cs, ops)


H = pauli_sum()
try:
    dev = qml.device(args.device, wires=n)
    diff = "adjoint"
except Exception:
    print(f"device {args.device} unavailable -> falling back to default.qubit (slower)")
    dev = qml.device("default.qubit", wires=n)
    diff = "backprop"


@qml.qnode(dev, interface="autograd", diff_method=diff)
def energy(params, bits):
    """<bits| U^T H U |bits>  with H = sum of Pauli terms (identity shift excluded)."""
    qml.BasisState(np.array(bits), wires=range(n))
    for l in range(L):
        for q in range(n):
            qml.RY(params[l, q], wires=q)
        for q in range(n - 1):                 # linear chain, no wrap-around
            qml.CNOT(wires=[q, q + 1])
    for q in range(n):
        qml.RY(params[L, q], wires=q)
    return qml.expval(H)


bits = [[int(b) for b in format(i, f"0{n}b")] for i in range(T)]


def sigma(params, i):
    return float(energy(params, bits[i])) + shift


def ky_fan(params):
    return sum(q_w[i] * (energy(params, bits[i]) + shift) for i in range(T))


def cost(p):
    return -ky_fan(p)


def train(p0, iters, lr0, lr1):
    p = np.array(p0, requires_grad=True)
    opt = qml.AdamOptimizer(lr0)
    t0 = time.time()
    for it in range(iters):
        opt.stepsize = lr0 * (lr1 / lr0) ** (it / iters)
        p, c = opt.step_and_cost(cost, p)
        if (it + 1) % 100 == 0 or it == 0:
            print(f"    iter {it+1:5d}  loss {-c:.4f} / {LOSS_OPT:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    return float(ky_fan(p)), onp.array(p)


print(f"{args.size}: {n} qubits, T={T}, layers={L}, params={(L+1)*n}, "
      f"CNOTs per circuit = {L*(n-1)}, simulator = {dev.name} ({diff})")
print(f"classical top-{T}: {onp.round(S_true, 4)}  |  optimal Ky Fan loss {LOSS_OPT:.4f}  "
      f"|  classical SVD time {classical_time*1e3:.1f} ms")

t_train = time.time()
if args.resume and os.path.exists(ckpt):
    p0 = onp.load(ckpt)
    assert p0.shape == (L + 1, n), f"{ckpt} shape {p0.shape} != {(L + 1, n)} (use the same --layers)"
    print(f"resuming from {ckpt} (lower learning rate)")
    best_loss, best_p = train(p0, args.iters, 0.03, 0.003)
else:
    best_loss, best_p = -1e9, None
    for r in range(args.restarts):
        rng = onp.random.default_rng(1000 * args.seed + 10 * n + r)
        p0 = rng.uniform(-onp.pi, onp.pi, (L + 1, n)) * [0.1, 0.3][r % 2]
        print(f"  restart {r}")
        loss, p = train(p0, args.iters, 0.1, 0.01)
        if loss > best_loss:
            best_loss, best_p = loss, p
train_time = time.time() - t_train

onp.save(ckpt, best_p)
sig = onp.array([sigma(np.array(best_p), i) for i in range(T)])
print(f"\nfinal loss {best_loss:.4f} / optimum {LOSS_OPT:.4f}  (gap {LOSS_OPT - best_loss:.4f})")
print(f"classical sigma : {onp.round(S_true, 4)}")
print(f"VQSVD sigma     : {onp.round(sig, 4)}")
print(f"mean abs error  : {onp.abs(sig - S_true).mean():.4f}   "
      f"mean rel error {onp.mean(onp.abs(sig - S_true) / S_true) * 100:.2f}%")
print(f"training time {train_time:.0f}s  vs classical SVD {classical_time*1e3:.1f} ms")
print(f"saved -> {ckpt}")