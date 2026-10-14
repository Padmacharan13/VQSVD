"""
vqsvd_train_chain.py
====================
Train the HARDWARE-FRIENDLY VQSVD ansatz on the ideal simulator.

Same as your vqsvd_<size>.py training (RY rotations + CNOTs, Ky Fan loss,
same test matrix), but the CNOTs form a LINEAR CHAIN (0-1, 1-2, ..., n-2 - n-1)
instead of a ring. The ring's wrap-around CNOT (n-1 -> 0) is not a neighbour
pair on IBM's heavy-hex chips, so the transpiler inserts SWAPs that roughly
triple the two-qubit gate count. A chain maps straight onto a path of qubits.

    python vqsvd_train_chain.py --size 32x32
    python vqsvd_train_chain.py --size 64x64
    python vqsvd_train_chain.py --size 32x32 --resume --iters 300   # keep improving

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
ap.add_argument("--iters", type=int, default=500, help="iterations per run")
ap.add_argument("--restarts", type=int, default=2, help="random restarts (fresh run only)")
ap.add_argument("--layers", type=int, default=None, help="override ansatz depth")
ap.add_argument("--resume", action="store_true", help="warm-start from the saved parameters")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

cfg = SIZES[args.size]
n, T = cfg["n"], cfg["T"]
L = args.layers or cfg["layers"]
ckpt = f"params_{args.size}_chain.npy"

M_np, terms, shift = build_matrix(n)
S_true = onp.sort(onp.linalg.eigvalsh(M_np))[::-1][:T]
M = np.array(M_np)
q_w = [float(T - i) for i in range(T)]
LOSS_OPT = float(sum(q_w[i] * S_true[i] for i in range(T)))

dev = qml.device("default.qubit", wires=n)


@qml.qnode(dev, interface="autograd", diff_method="backprop")
def state(params, bits):
    qml.BasisState(np.array(bits), wires=range(n))
    for l in range(L):
        for q in range(n):
            qml.RY(params[l, q], wires=q)
        for q in range(n - 1):                 # linear chain, no wrap-around
            qml.CNOT(wires=[q, q + 1])
    for q in range(n):
        qml.RY(params[L, q], wires=q)
    return qml.state()


bits = [[int(b) for b in format(i, f"0{n}b")] for i in range(T)]


def ky_fan(params):
    tot = 0.0
    for i in range(T):
        u = state(params, bits[i])
        tot = tot + q_w[i] * np.real(np.sum(np.conj(u) * (M @ u)))
    return tot


def cost(p):
    return -ky_fan(p)


def train(p0, iters, lr0, lr1):
    p = np.array(p0, requires_grad=True)
    opt = qml.AdamOptimizer(lr0)
    t0 = time.time()
    for it in range(iters):
        opt.stepsize = lr0 * (lr1 / lr0) ** (it / iters)
        p, c = opt.step_and_cost(cost, p)
        if (it + 1) % 50 == 0:
            print(f"    iter {it+1:4d}  loss {-c:.4f} / {LOSS_OPT:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    return float(ky_fan(p)), onp.array(p)


print(f"{args.size}: {n} qubits, T={T}, layers={L}, params={(L+1)*n}, "
      f"CNOTs per circuit = {L*(n-1)} (ring would be {L*n})")
print(f"classical top-{T}: {onp.round(S_true, 4)}  |  optimal Ky Fan loss {LOSS_OPT:.4f}")

if args.resume and os.path.exists(ckpt):
    p0 = onp.load(ckpt)
    assert p0.shape == (L + 1, n), f"{ckpt} shape {p0.shape} != {(L + 1, n)}"
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

onp.save(ckpt, best_p)
sig = []
for i in range(T):
    u = state(np.array(best_p), bits[i])
    sig.append(float(np.real(np.sum(np.conj(u) * (M @ u)))))
sig = onp.array(sig)
print(f"\nfinal loss {best_loss:.4f} / optimum {LOSS_OPT:.4f}  (gap {LOSS_OPT - best_loss:.4f})")
print(f"classical sigma : {onp.round(S_true, 4)}")
print(f"VQSVD sigma     : {onp.round(sig, 4)}")
print(f"mean abs error  : {onp.abs(sig - S_true).mean():.4f}   "
      f"mean rel error {onp.mean(onp.abs(sig - S_true) / S_true) * 100:.2f}%")
print(f"saved -> {ckpt}")