# VQSVD on IBM Quantum hardware: without vs. with error mitigation

Generated 2026-10-16T16:49:59

## Summary

| Matrix | Mitigation | MAE without (raw) | MAE with | Change |
|---|---|---|---|---|
| 512x512 | Readout only | 4.5574 | 4.2876 | 5.9% better |
| 512x512 | ZNE only | 4.5574 | 2.6269 | 42.4% better |
| 512x512 | ZNE + readout | 4.5574 | 2.2870 | 49.8% better |
| 1024x1024 | Readout only | 7.6974 | 7.4807 | 2.8% better |
| 1024x1024 | ZNE only | 7.6974 | 5.1489 | 33.1% better |
| 1024x1024 | ZNE + readout | 7.6974 | 4.8506 | 37.0% better |

MAE = mean absolute error of the extracted singular values vs. classical SVD (lower is better).

## 512x512 (chain ansatz, 9 qubits)

- Backend: **ibm_marrakesh**, physical qubits [9, 10, 11, 12, 13, 14, 15, 19, 35], 2026-10-16T16:39:29
- Transpiled circuit: depth 269, **240 two-qubit gates** (native CNOTs 240); optimization level 3
- Device error rates used: median 2-qubit gate 0.193%, median readout 1.709%
- Shots 4000, ZNE scales [1, 3], mitigation mode `custom`
- Time: classical SVD 90.4897 ms vs hardware evaluation 421.4 s wall-clock, 20.0 s QPU
- Jobs: readout calibration: `db90bqob3g5c73eh4ifg`, ZNE scales [1, 3]: `db90bre94fic73eu8e30`

| Condition | sigma[0] | sigma[1] | sigma[2] | MAE vs classical | MAE vs ideal | Signal kept |
|---|---|---|---|---|---|---|
| Classical SVD | 25.8886 | 22.6299 | 22.4401 | - | - | - |
| Ideal (no noise) | 25.8851 | 22.4280 | 22.0246 | 0.2070 | 0.0000 | 100.0% |
| Hardware, NO mitigation | 20.4082 | 18.2840 | 18.5943 | 4.5574 | 4.3504 | 63.0% |
| Readout only | 20.7377 | 18.5220 | 18.8361 | 4.2876 | 4.0806 | 65.3% |
| ZNE only | 22.8125 | 19.8292 | 20.4362 | 2.6269 | 2.4199 | 79.5% |
| ZNE + readout | 23.2336 | 20.1241 | 20.7399 | 2.2870 | 2.0800 | 82.4% |

> **Result:** best mitigation was *ZNE + readout*: MAE 4.5574 -> 2.2870 (improved by 49.8%).

## 1024x1024 (chain ansatz, 10 qubits)

- Backend: **ibm_marrakesh**, physical qubits [8, 9, 10, 11, 12, 13, 14, 15, 19, 35], 2026-10-16T16:44:44
- Transpiled circuit: depth 320, **324 two-qubit gates** (native CNOTs 324); optimization level 3
- Device error rates used: median 2-qubit gate 0.214%, median readout 1.880%
- Shots 4000, ZNE scales [1, 3], mitigation mode `custom`
- Time: classical SVD 3389.1745 ms vs hardware evaluation 493.4 s wall-clock, 20.0 s QPU
- Jobs: readout calibration: `db90e9ob3g5c73eh4mkg`, ZNE scales [1, 3]: `db90eagb3g5c73eh4mmg`

| Condition | sigma[0] | sigma[1] | sigma[2] | MAE vs classical | MAE vs ideal | Signal kept |
|---|---|---|---|---|---|---|
| Classical SVD | 32.0232 | 29.0436 | 28.9337 | - | - | - |
| Ideal (no noise) | 32.0125 | 28.9004 | 28.6351 | 0.1508 | 0.0000 | 100.0% |
| Hardware, NO mitigation | 23.0760 | 21.8596 | 21.9726 | 7.6974 | 7.5466 | 49.8% |
| Readout only | 23.3274 | 22.0665 | 22.1644 | 7.4807 | 7.3299 | 51.2% |
| ZNE only | 25.9576 | 24.2161 | 24.3801 | 5.1489 | 4.9981 | 66.8% |
| ZNE + readout | 26.3067 | 24.5009 | 24.6411 | 4.8506 | 4.6998 | 68.8% |

> **Result:** best mitigation was *ZNE + readout*: MAE 7.6974 -> 4.8506 (improved by 37.0%).

![report chart](ibm_hardware_report.png)
