"""
vqsvd_ibm_hardware.py   (v2)
============================
Evaluate TRAINED VQSVD circuits on IBM Quantum hardware (or a local noisy copy
of an IBM chip) WITHOUT and WITH error mitigation, and write a report.

Typical use for 32x32 and 64x64
-------------------------------
  # 0. one-time: pip install qiskit qiskit-ibm-runtime numpy matplotlib
  #    and save your IBM API key (see SETUP below)
  # 1. train on the simulator (you already have these for the ring ansatz):
  python vqsvd_32x32.py --skip-noise          # -> params_32x32.npy   (ring)
  python vqsvd_64x64.py --skip-noise          # -> params_64x64.npy   (ring)
  #    hardware-friendly variant (recommended, ~3x fewer two-qubit gates):
  python vqsvd_train_chain.py --size 32x32    # -> params_32x32_chain.npy
  python vqsvd_train_chain.py --size 64x64    # -> params_64x64_chain.npy
  # 2. exact check that Qiskit reproduces the simulator (no account needed):
  python vqsvd_ibm_hardware.py --sizes 32x32 64x64 --entangler chain --mode ideal
  # 3. look before you spend QPU time (needs the account, submits NOTHING):
  python vqsvd_ibm_hardware.py --sizes 32x32 64x64 --entangler chain --mode hardware --dry-run
  # 4. run on the real chip, writes ibm_hardware_report.md / .png:
  python vqsvd_ibm_hardware.py --sizes 32x32 64x64 --entangler chain --mode hardware
  # 5. if the script was interrupted while waiting in the queue:
  python vqsvd_ibm_hardware.py --sizes 32x32 64x64 --entangler chain --mode hardware --retrieve
  # 6. rebuild the report any time from the saved results (no hardware needed):
  python vqsvd_ibm_hardware.py --sizes 32x32 64x64 --entangler chain --report-only

What one run produces (mitigation = custom, the default)
  Raw hardware, readout-only, ZNE-only and ZNE+readout values come from the
  SAME hardware session, so the comparison is fair (same calibration/drift).
  "raw" is the unmitigated baseline.

Modes:  --mode ideal     exact check, no noise, no account
        --mode fake      local noisy simulation of a real IBM chip (--fake FakeFez)
        --mode hardware  real QPU
Mitigation: --mitigation custom (default; your ZNE + readout, run on raw execution)
            --mitigation raw    (baseline only)
            --mitigation ibm    (IBM built-in ZNE + TREX + DD + twirling; costs more QPU time)

SETUP (once)
  Create a free account at https://quantum.cloud.ibm.com, copy the API key, then:
      from qiskit_ibm_runtime import QiskitRuntimeService
      QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                        token="YOUR_API_KEY", set_as_default=True)
  Open Plan allowance: 10 minutes of QPU time per 28 days. Every job's QPU
  seconds are printed and stored. Results are written to disk the moment a job
  finishes, so a crash during post-processing never wastes QPU time.
"""
import argparse
import csv
import datetime
import json
import os
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

SEED = 11   # must match the training scripts so the SAME matrix is rebuilt
SIZES = {
    "4x4":     dict(n=2, T=2, layers=4),
    "8x8":     dict(n=3, T=3, layers=6),
    "16x16":   dict(n=4, T=3, layers=8),
    "32x32":   dict(n=5, T=3, layers=12),
    "64x64":   dict(n=6, T=3, layers=16),
    "128x128": dict(n=7, T=3, layers=22),
    "512x512": dict(n=9, T=3, layers=30),
    "1024x1024": dict(n=10, T=3, layers=36),
}


# ----------------------------------------------------------------------
# Test matrix (identical to the training scripts, rebuilt from the same seed)
# ----------------------------------------------------------------------
def build_matrix(n):
    X = np.array([[0, 1], [1, 0]])
    Z = np.diag([1, -1])
    I2 = np.eye(2)

    def op_at(op, q):
        mats = [I2] * n
        mats[q] = op
        m = mats[0]
        for a in mats[1:]:
            m = np.kron(m, a)
        return m

    rng = np.random.default_rng(SEED + n)
    M = np.zeros((2 ** n, 2 ** n))
    terms = []     # (coef, pauli_type, qubits)
    for q in range(n):
        c = rng.uniform(0.5, 1.5)
        M += c * op_at(X, q); terms.append((c, "X", [q]))
        c = rng.uniform(0.2, 0.8)
        M += c * op_at(Z, q); terms.append((c, "Z", [q]))
    for q in range(n if n > 2 else 1):
        c = rng.uniform(0.5, 1.5)
        M += c * (op_at(Z, q) @ op_at(Z, (q + 1) % n))
        terms.append((c, "ZZ", [q, (q + 1) % n]))
    shift = abs(np.linalg.eigvalsh(M).min()) + 0.5
    M += shift * np.eye(2 ** n)
    return M, terms, shift


# ----------------------------------------------------------------------
# Circuits: identical to the PennyLane ansatz (RY layers + CNOT entangler)
# ----------------------------------------------------------------------
def entangle(qc, n, entangler, inverse=False):
    if entangler == "ring":
        pairs = [(q, (q + 1) % n) for q in range(n)]
    else:                                   # chain: no wrap-around
        pairs = [(q, q + 1) for q in range(n - 1)]
    for a, b in (reversed(pairs) if inverse else pairs):
        qc.cx(a, b)


def forward_block(qc, p, n, L, ent):
    for l in range(L):
        for q in range(n):
            qc.ry(float(p[l, q]), q)
        entangle(qc, n, ent)
    for q in range(n):
        qc.ry(float(p[L, q]), q)


def inverse_block(qc, p, n, L, ent):
    for q in range(n):
        qc.ry(-float(p[L, q]), q)
    for l in reversed(range(L)):
        entangle(qc, n, ent, inverse=True)
        for q in range(n):
            qc.ry(-float(p[l, q]), q)


def build_circuit(p, bits, n, L, scale=1, ent="ring"):
    """Basis state |bits>, then the ansatz. scale>1 = unitary folding (ZNE):
    U -> U (U^-1 U)^k, with barriers so the transpiler cannot cancel the pairs."""
    qc = QuantumCircuit(n)
    for q, b in enumerate(bits):
        if b:
            qc.x(q)
    qc.barrier()
    forward_block(qc, p, n, L, ent)
    for _ in range((scale - 1) // 2):
        qc.barrier()
        inverse_block(qc, p, n, L, ent)
        qc.barrier()
        forward_block(qc, p, n, L, ent)
    return qc


def calibration_circuit(n, ones):
    qc = QuantumCircuit(n)
    if ones:
        for q in range(n):
            qc.x(q)
    return qc


def unit_ops(terms, n):
    return [SparsePauliOp.from_sparse_list([(pt, qs, 1.0)], num_qubits=n) for _, pt, qs in terms]


def total_op(terms, n):
    return SparsePauliOp.from_sparse_list([(pt, qs, c) for c, pt, qs in terms], num_qubits=n)


# ----------------------------------------------------------------------
# Backend helpers
# ----------------------------------------------------------------------
def get_service(args):
    from qiskit_ibm_runtime import QiskitRuntimeService
    kw = dict(channel="ibm_quantum_platform")
    if args.token:
        kw["token"] = args.token
    if args.instance:
        kw["instance"] = args.instance
    try:
        return QiskitRuntimeService(**kw)
    except Exception as e:
        raise SystemExit(
            f"Could not connect to IBM Quantum ({type(e).__name__}: {e}).\n"
            "Save your account once:\n"
            "  from qiskit_ibm_runtime import QiskitRuntimeService\n"
            "  QiskitRuntimeService.save_account(channel='ibm_quantum_platform',\n"
            "                                    token='YOUR_API_KEY', set_as_default=True)\n"
            "or pass --token YOUR_API_KEY (and --instance if you have several).")


def get_backend(args, n, service):
    if args.mode == "fake":
        from qiskit_ibm_runtime import fake_provider
        return getattr(fake_provider, args.fake)()
    if args.backend:
        return service.backend(args.backend)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=n)


def transpile_all(circuits, backend, opt_level, initial_layout=None):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(backend=backend, optimization_level=opt_level,
                                      seed_transpiler=1, initial_layout=initial_layout)
    return [pm.run(c) for c in circuits]


def physical_qubits(isa):
    return list(isa.layout.initial_index_layout(filter_ancillas=True))


def backend_error_summary(backend, phys):
    tgt = backend.target
    two_q, ro = [], []
    for op in ("cz", "ecr", "cx"):
        if op in tgt.operation_names:
            for qargs, props in tgt[op].items():
                if props is not None and props.error is not None and all(q in phys for q in qargs):
                    two_q.append(props.error)
    if "measure" in tgt.operation_names:
        for q in phys:
            props = tgt["measure"].get((q,))
            if props is not None and props.error is not None:
                ro.append(props.error)
    med = lambda v: float(np.median(v)) if v else float("nan")
    return med(two_q), med(ro)


def make_estimator(backend, args):
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    est = Estimator(mode=backend)
    est.options.default_shots = args.shots
    if args.mode == "fake":
        return est          # runtime-side mitigation options do not apply locally
    est.options.resilience_level = 0
    if args.mitigation == "ibm":
        try:
            est.options.dynamical_decoupling.enable = True
            est.options.twirling.enable_gates = True
            est.options.twirling.enable_measure = True
            est.options.resilience.measure_mitigation = True
            est.options.resilience.zne_mitigation = True
            est.options.resilience.zne.noise_factors = list(args.scales)
            est.options.resilience.zne.extrapolator = "linear"
        except Exception as e:     # option names shift between runtime versions
            print(f"  WARNING: could not set an IBM mitigation option ({e}); "
                  f"check your qiskit-ibm-runtime version.")
    return est


# ----------------------------------------------------------------------
# State files (so nothing is ever lost between submit / wait / post-process)
# ----------------------------------------------------------------------
def tag(size, args):
    return f"{size}_{args.entangler}_{args.mitigation}"


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def evs_of(res, k):
    return np.asarray(res[k].data.evs, dtype=float).tolist()


# ----------------------------------------------------------------------
# Post-processing (pure function of saved data -> can be re-run offline)
# ----------------------------------------------------------------------
def postprocess(meta, groups):
    T, shift = meta["T"], meta["shift"]
    ideal = np.array(meta["ideal"])
    results = {"ideal": ideal}
    if meta["mitigation"] in ("raw", "ibm"):
        vals = np.array([float(np.asarray(v)) for v in groups[0]["evs"]]) + shift
        results["raw" if meta["mitigation"] == "raw" else "ibm_mitigated"] = vals
        return results
    cal = groups[0]["evs"]
    z0, z1 = np.array(cal[0]), np.array(cal[1])
    d = (z0 - z1) / 2.0                    # = 1 - p(0->1) - p(1->0) per qubit
    meta["readout_damping"] = d.tolist()
    coefs = np.array(meta["coefs"])
    damp = np.array([np.prod([d[q] for q in sup]) for sup in meta["supports"]])
    scales = meta["scales"]
    raw_pts = {i: [] for i in range(T)}
    corr_pts = {i: [] for i in range(T)}
    for k, (s, i) in enumerate(meta["keys"]):
        ev = np.array(groups[1]["evs"][k]).reshape(-1)
        raw_pts[i].append(shift + float(np.dot(coefs, ev)))
        corr_pts[i].append(shift + float(np.dot(coefs, ev / damp)))
    for name in ("raw", "readout_only", "zne_only", "combined"):
        results[name] = np.zeros(T)
    for i in range(T):
        results["raw"][i] = raw_pts[i][0]
        results["readout_only"][i] = corr_pts[i][0]
        if len(scales) > 1:
            results["zne_only"][i] = np.polyfit(scales, raw_pts[i], 1)[1]
            results["combined"][i] = np.polyfit(scales, corr_pts[i], 1)[1]
        else:
            results["zne_only"][i], results["combined"][i] = raw_pts[i][0], corr_pts[i][0]
    meta["raw_points"] = {str(i): raw_pts[i] for i in range(T)}
    meta["corr_points"] = {str(i): corr_pts[i] for i in range(T)}
    return results


def summarize(meta, results):
    S, ideal, shift = np.array(meta["S_true"]), np.array(meta["ideal"]), meta["shift"]
    rows = {}
    for k, v in results.items():
        v = np.asarray(v)
        sig_ideal = np.where(np.abs(ideal - shift) > 1e-9, ideal - shift, np.nan)
        rows[k] = dict(values=v.tolist(),
                       mae_vs_classical=float(np.abs(v - S).mean()),
                       mae_vs_ideal=float(np.abs(v - ideal).mean()),
                       signal_survival=float(np.nanmean((v - shift) / sig_ideal)))
    return rows


def print_results(meta, rows):
    T = meta["T"]
    S = meta["S_true"]
    print("\n" + "=" * 78)
    print(f"RESULTS -- {meta['size']} ({meta['entangler']} ansatz) on {meta['backend']}  "
          f"[mode={meta['mode']}, mitigation={meta['mitigation']}]")
    print("=" * 78)
    print(f"{'condition':<15}" + "".join(f"sigma[{i}]".rjust(10) for i in range(T)) +
          "   MAE vs classical   MAE vs ideal   signal kept")
    print(f"{'classical':<15}" + "".join(f"{S[i]:10.4f}" for i in range(T)))
    for k, r in rows.items():
        print(f"{k:<15}" + "".join(f"{r['values'][i]:10.4f}" for i in range(T)) +
              f"   {r['mae_vs_classical']:14.4f}   {r['mae_vs_ideal']:11.4f}   {r['signal_survival']*100:8.1f}%")
    print("(signal kept = fraction of the circuit's non-identity signal that survived; "
          "100% = exact, 0% = pure noise)")


# ----------------------------------------------------------------------
# One matrix size
# ----------------------------------------------------------------------
def run_size(size, args, service):
    cfg = SIZES[size]
    n, T, L = cfg["n"], cfg["T"], cfg["layers"]
    ent = args.entangler
    pfile = args.params or (f"params_{size}.npy" if ent == "ring" else f"params_{size}_chain.npy")
    spath = f"ibm_{tag(size, args)}_state.json"
    rpath = f"ibm_{tag(size, args)}_result.json"

    if args.retrieve or args.from_saved:
        if not os.path.exists(spath):
            raise SystemExit(f"{spath} not found - nothing to retrieve/re-process for {size}.")
        st = json.load(open(spath))
        meta, groups = st["meta"], st["groups"]
        print(f"\n[{size}] loaded saved state {spath}")
    else:
        if not os.path.exists(pfile):
            hint = (f"python vqsvd_train_chain.py --size {size}" if ent == "chain"
                    else f"python vqsvd_{size}.py --skip-noise")
            raise SystemExit(f"{pfile} not found - train first:  {hint}")
        p = np.load(pfile)
        if p.ndim != 2 or p.shape[1] != n or p.shape[0] < 2:
            raise SystemExit(f"{pfile} has shape {p.shape}; expected (layers+1, {n}) for {size}")
        L = p.shape[0] - 1                      # depth comes from the trained parameters

        M, terms, shift = build_matrix(n)
        S_true = np.sort(np.linalg.svd(M, compute_uv=False))[::-1][:T]
        times = []
        for _ in range(5):                      # classical SVD timing, best of 5
            t_start = time.perf_counter()
            np.linalg.svd(M)
            times.append(time.perf_counter() - t_start)
        bits = [[int(b) for b in format(i, f"0{n}b")] for i in range(T)]
        H = total_op(terms, n)
        units = unit_ops(terms, n)
        coefs = np.array([c for c, _, _ in terms])
        supports = [qs for _, _, qs in terms]
        ideal = np.array([Statevector(build_circuit(p, bits[i], n, L, 1, ent)).expectation_value(H).real
                          + shift for i in range(T)])
        print(f"\n[{size}] {n} qubits | top-{T} | {ent} ansatz | mode={args.mode} | params {pfile}")
        print(f"  classical SVD : {np.round(S_true, 4)}   ({min(times)*1e3:.4f} ms)")
        print(f"  exact (ideal) : {np.round(ideal, 4)}   <- trained circuit, no noise")
        if args.mode == "ideal":
            print(f"  mean abs error vs classical: {np.abs(ideal - S_true).mean():.4f}")
            print("  (these should equal the sigma printed by the training script)")
            return None

        scales = list(args.scales) if args.mitigation in ("custom", "ibm") else [1]
        backend = get_backend(args, n, service)
        c1 = [build_circuit(p, bits[i], n, L, 1, ent) for i in range(T)]
        isa1 = transpile_all(c1, backend, args.opt_level)
        layout = physical_qubits(isa1[0])
        isa1 = transpile_all(c1, backend, args.opt_level, layout)
        e2q, ero = backend_error_summary(backend, layout)
        ops = isa1[0].count_ops()
        n2q = int(sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx")))
        native = L * (n if ent == "ring" else n - 1)
        print(f"  backend       : {backend.name} ({backend.num_qubits} qubits), physical qubits {layout}")
        print(f"  transpiled    : depth {isa1[0].depth()}, {n2q} two-qubit gates "
              f"(circuit has {native} CNOTs; extra = routing SWAPs)")
        print(f"  device errors : median 2-qubit gate {e2q*100:.3f}%, median readout {ero*100:.3f}%")
        surv = {s: (1 - e2q) ** (n2q * s) for s in sorted(set([1] + scales))}
        print("  rough gate-noise survival (1-e)^gates: " +
              ", ".join(f"{s}x: {v*100:.1f}%" for s, v in surv.items()))
        if surv[1] < 0.10:
            print("  WARNING: under ~10% of the signal is expected to survive even at 1x. Raw hardware "
                  "will be mostly noise and mitigation cannot recover erased signal. Consider "
                  "--entangler chain (far fewer gates) before spending QPU time.")
        n_pubs = T * len(scales) + (2 if args.mitigation == "custom" else 0)
        print(f"  plan          : {n_pubs} estimator pubs x {args.shots} shots "
              f"(~2 measurement bases each), scales {scales}")
        meta = dict(size=size, n=n, T=T, L=L, entangler=ent, mitigation=args.mitigation, mode=args.mode,
                    shots=args.shots, scales=scales, shift=float(shift), S_true=S_true.tolist(),
                    ideal=ideal.tolist(), classical_time=min(times), coefs=coefs.tolist(),
                    supports=supports, params_file=pfile, backend=backend.name, layout=layout,
                    depth=int(isa1[0].depth()), n2q=n2q, native_cnots=native, e2q=e2q, ero=ero,
                    opt_level=args.opt_level, timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                    keys=[(s, i) for s in scales for i in range(T)])
        if args.dry_run:
            print("  --dry-run: nothing submitted.")
            return None

        # ---- build the job groups ----
        est = make_estimator(backend, args)
        job_specs = []          # (name, pubs)
        if args.mitigation == "custom":
            cal = transpile_all([calibration_circuit(n, False), calibration_circuit(n, True)],
                                backend, args.opt_level, layout)
            zops = [SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], num_qubits=n) for q in range(n)]
            job_specs.append(("readout calibration",
                              [(c, [z.apply_layout(c.layout) for z in zops]) for c in cal]))
            pubs = []
            for s in scales:
                circs = transpile_all([build_circuit(p, bits[i], n, L, s, ent) for i in range(T)],
                                      backend, args.opt_level, layout)
                for i in range(T):
                    pubs.append((circs[i], [u.apply_layout(circs[i].layout) for u in units]))
            job_specs.append(("ZNE scales " + str(scales), pubs))
        else:
            job_specs.append((args.mitigation, [(isa1[i], H.apply_layout(isa1[i].layout)) for i in range(T)]))

        groups, jobs = [], []
        for name, pubs in job_specs:
            job = est.run(pubs)
            jid = job.job_id() if hasattr(job, "job_id") else "local"
            print(f"  submitted [{name}] job {jid}", flush=True)
            groups.append(dict(name=name, job_id=jid, evs=None, wall=None, qsec=None,
                               t_submit=time.time()))
            jobs.append(job)
        save_json(spath, dict(meta=meta, groups=groups))
        print(f"  state saved -> {spath}   (use --retrieve if this script is interrupted)")
        for g, job in zip(groups, jobs):
            res = job.result()
            g["wall"] = time.time() - g["t_submit"]
            try:
                g["qsec"] = job.usage()
            except Exception:
                g["qsec"] = None
            g["evs"] = [evs_of(res, k) for k in range(len(res))]
            save_json(spath, dict(meta=meta, groups=groups))     # saved the moment results arrive
            print(f"  [{g['name']}] done: wall {g['wall']:.1f}s"
                  + (f", QPU {g['qsec']}s" if g["qsec"] is not None else ""))

    if args.retrieve:
        service = service or get_service(args)
        for g in groups:
            if g["evs"] is None:
                job = service.job(g["job_id"])
                print(f"  waiting for job {g['job_id']} ({g['name']}) ...", flush=True)
                res = job.result()
                try:
                    g["qsec"] = job.usage()
                except Exception:
                    pass
                g["evs"] = [evs_of(res, k) for k in range(len(res))]
                save_json(spath, dict(meta=meta, groups=groups))
    if any(g["evs"] is None for g in groups):
        raise SystemExit("some jobs have no results yet - run again with --retrieve")

    # ---- post-process and save ----
    results = postprocess(meta, groups)
    rows = summarize(meta, results)
    print_results(meta, rows)
    total_wall = sum(g["wall"] or 0 for g in groups)
    total_q = sum(g["qsec"] or 0 for g in groups)
    print(f"\nTime: classical SVD {meta['classical_time']*1e3:.4f} ms  vs  hardware evaluation "
          f"{total_wall:.1f} s wall-clock" + (f" ({total_q:.1f} s QPU)" if total_q else "") +
          "  [evaluation only; training was done on the simulator]")
    save_json(rpath, dict(meta=meta, rows=rows, groups=[{k: v for k, v in g.items() if k != "evs"} for g in groups],
                          total_wall=total_wall, total_qpu=total_q))
    print(f"Saved -> {rpath}")
    return rpath


# ----------------------------------------------------------------------
# Report (Markdown + figure) from saved result files
# ----------------------------------------------------------------------
def build_report(paths, out="ibm_hardware_report"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for pth in paths:
        if pth and os.path.exists(pth):
            runs.append(json.load(open(pth)))
        else:
            print(f"  (report) missing {pth} - skipped")
    if not runs:
        print("No result files found - nothing to report.")
        return
    labels = {"ideal": "Ideal (no noise)", "raw": "Hardware, NO mitigation", "readout_only": "Readout only",
              "zne_only": "ZNE only", "combined": "ZNE + readout", "ibm_mitigated": "IBM built-in mitigation"}
    colors = {"ideal": "#2e8b57", "raw": "#c0392b", "readout_only": "#3b6fb6", "zne_only": "#e67e22",
              "combined": "#8e44ad", "ibm_mitigated": "#16a085"}

    md = ["# VQSVD on IBM Quantum hardware: without vs. with error mitigation\n",
          f"Generated {datetime.datetime.now().isoformat(timespec='seconds')}\n"]
    summary = []
    for r in runs:
        m, rows = r["meta"], r["rows"]
        raw = rows.get("raw", {}).get("mae_vs_classical")
        for k, v in rows.items():
            if k in ("ideal", "raw") or raw is None:
                continue
            summary.append((m["size"], k, raw, v["mae_vs_classical"], (raw - v["mae_vs_classical"]) / raw * 100))
    md.append("## Summary\n")
    md.append("| Matrix | Mitigation | MAE without (raw) | MAE with | Change |\n|---|---|---|---|---|")
    for size, k, raw, v, ch in summary:
        verdict = "better" if ch > 0 else "WORSE"
        md.append(f"| {size} | {labels.get(k, k)} | {raw:.4f} | {v:.4f} | {abs(ch):.1f}% {verdict} |")
    md.append("\nMAE = mean absolute error of the extracted singular values vs. classical SVD (lower is better).\n")

    fig, axes = plt.subplots(2, len(runs), figsize=(5.2 * len(runs), 8), squeeze=False)
    for c, r in enumerate(runs):
        m, rows = r["meta"], r["rows"]
        md.append(f"## {m['size']} ({m['entangler']} ansatz, {m['n']} qubits)\n")
        md.append(f"- Backend: **{m['backend']}**, physical qubits {m['layout']}, {m['timestamp']}")
        md.append(f"- Transpiled circuit: depth {m['depth']}, **{m['n2q']} two-qubit gates** "
                  f"(native CNOTs {m['native_cnots']}); optimization level {m['opt_level']}")
        md.append(f"- Device error rates used: median 2-qubit gate {m['e2q']*100:.3f}%, "
                  f"median readout {m['ero']*100:.3f}%")
        md.append(f"- Shots {m['shots']}, ZNE scales {m['scales']}, mitigation mode `{m['mitigation']}`")
        md.append(f"- Time: classical SVD {m['classical_time']*1e3:.4f} ms vs hardware evaluation "
                  f"{r['total_wall']:.1f} s wall-clock" + (f", {r['total_qpu']:.1f} s QPU" if r["total_qpu"] else ""))
        md.append(f"- Jobs: " + ", ".join(f"{g['name']}: `{g['job_id']}`" for g in r["groups"]) + "\n")
        T = m["T"]
        md.append("| Condition | " + " | ".join(f"sigma[{i}]" for i in range(T)) +
                  " | MAE vs classical | MAE vs ideal | Signal kept |")
        md.append("|---|" + "---|" * (T + 3))
        md.append("| Classical SVD | " + " | ".join(f"{x:.4f}" for x in m["S_true"]) + " | - | - | - |")
        for k, v in rows.items():
            md.append(f"| {labels.get(k, k)} | " + " | ".join(f"{x:.4f}" for x in v["values"]) +
                      f" | {v['mae_vs_classical']:.4f} | {v['mae_vs_ideal']:.4f} | {v['signal_survival']*100:.1f}% |")
        raw = rows.get("raw")
        if raw:
            best = min((k for k in rows if k not in ("ideal", "raw")),
                       key=lambda k: rows[k]["mae_vs_classical"], default=None)
            md.append("")
            if raw["signal_survival"] < 0.10:
                md.append(f"> **Note:** only {raw['signal_survival']*100:.1f}% of the circuit's signal survived on "
                          f"raw hardware. Mitigation cannot recover signal that noise has already erased; "
                          f"this is the depth limit of current hardware for this circuit.")
            if best:
                ch = (raw["mae_vs_classical"] - rows[best]["mae_vs_classical"]) / raw["mae_vs_classical"] * 100
                md.append(f"> **Result:** best mitigation was *{labels.get(best, best)}*: MAE "
                          f"{raw['mae_vs_classical']:.4f} -> {rows[best]['mae_vs_classical']:.4f} "
                          f"({'improved' if ch > 0 else 'worsened'} by {abs(ch):.1f}%).")
        md.append("")

        keys = list(rows)
        ax = axes[0, c]
        ax.bar(range(len(keys)), [rows[k]["mae_vs_classical"] for k in keys],
               color=[colors.get(k, "gray") for k in keys], edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([labels.get(k, k).replace(" ", "\n", 1) for k in keys], fontsize=7)
        ax.set_ylabel("MAE vs classical SVD")
        ax.set_title(f"{m['size']} on {m['backend']}\n({m['n2q']} two-qubit gates)", fontsize=10)
        for i, k in enumerate(keys):
            ax.text(i, rows[k]["mae_vs_classical"], f"{rows[k]['mae_vs_classical']:.3f}",
                    ha="center", va="bottom", fontsize=7)
        ax2 = axes[1, c]
        ax2.bar(range(len(keys)), [rows[k]["signal_survival"] * 100 for k in keys],
                color=[colors.get(k, "gray") for k in keys], edgecolor="black", linewidth=0.6)
        ax2.axhline(100, color="k", ls="--", lw=0.8)
        ax2.set_xticks(range(len(keys)))
        ax2.set_xticklabels([labels.get(k, k).replace(" ", "\n", 1) for k in keys], fontsize=7)
        ax2.set_ylabel("Signal kept (%)  (100 = exact)")
    fig.suptitle("VQSVD on IBM hardware: raw vs. mitigated", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out + ".png", dpi=180)
    plt.close(fig)
    md.append(f"![report chart]({os.path.basename(out)}.png)\n")
    with open(out + ".md", "w") as f:
        f.write("\n".join(md))
    print(f"\nReport written -> {out}.md  and  {out}.png")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", "--size", nargs="+", required=True, choices=list(SIZES))
    ap.add_argument("--entangler", choices=["ring", "chain"], default="ring")
    ap.add_argument("--params", default=None, help="parameter file (single size only)")
    ap.add_argument("--mode", choices=["ideal", "fake", "hardware"], default="ideal")
    ap.add_argument("--mitigation", choices=["custom", "raw", "ibm"], default="custom")
    ap.add_argument("--backend", default=None, help="IBM backend name, e.g. ibm_fez (default: least busy)")
    ap.add_argument("--fake", default="FakeFez", help="fake backend class for --mode fake")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 3],
                    help="ZNE noise factors (default 1 3; use 1 3 5 to match a 3-point simulator study)")
    ap.add_argument("--opt-level", type=int, default=3, choices=[0, 1, 2, 3], help="transpiler optimization level")
    ap.add_argument("--token", default=None)
    ap.add_argument("--instance", default=None)
    ap.add_argument("--dry-run", action="store_true", help="transpile and report cost; submit nothing")
    ap.add_argument("--retrieve", action="store_true", help="fetch results of previously submitted jobs")
    ap.add_argument("--from-saved", action="store_true", help="re-run post-processing from saved raw data")
    ap.add_argument("--report-only", action="store_true", help="only (re)build the report from saved results")
    ap.add_argument("--report", default="ibm_hardware_report", help="report file prefix")
    args = ap.parse_args()

    service = None
    if args.mode == "hardware" and not args.report_only and not args.from_saved:
        service = get_service(args)
    paths = []
    for size in args.sizes:
        if args.report_only:
            paths.append(f"ibm_{tag(size, args)}_result.json")
        else:
            paths.append(run_size(size, args, service))
    if (args.report_only or args.mode != "ideal") and not args.dry_run:
        build_report(paths, args.report)


if __name__ == "__main__":
    main()