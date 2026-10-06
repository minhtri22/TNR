
## Run 20261006_144648
- Date: 2026-10-06T14:46:51.143516
- Experiment: TNR minimal first experiment (A/B/C/D comparison)
- Config: HIDDEN=24, N_ITER=3, EPOCHS=80, lambda=0.01
- Results MSE: A=0.4045, B=0.4032, C=0.3996, D=0.4013
- Params: A=4108, B=10108, C=11380, D=24548
- Route fidelity D: 0.04332610676178064
- Outcome: **UNRESOLVED**
- Notes: Synthetic task with 3 case families variable dependency paths. Equal param/compute approximately controlled via width and iterations.

## Run 20261006_144921
- Date: 2026-10-06T14:49:25.113612
- Experiment: TNR minimal first experiment (A/B/C/D comparison)
- Config: HIDDEN=32, N_ITER=4, EPOCHS=60, lambda=0.001
- Results MSE: A=0.3954, B=0.3994, C=0.3985, D=0.3951
- Params: A=7012, B=17572, C=19524, D=35764
- Route fidelity D: 0.04584528257873432
- Outcome: **UNRESOLVED**
- Notes: Synthetic task with 3 case families variable dependency paths. Equal param/compute approximately controlled via width and iterations.

## Run 20261006_150521
- Date: 2026-10-06T15:05:25.142702
- Experiment: TNR v2 – kế hoạch run tiếp theo (capacity↑, param balance, top-K, thêm E/F, route F1)
- Config: HIDDEN=24, N_ITER=3, TOP_K=3, λ=0.0005, EPOCHS=30
- Results MSE: A=0.1728, B=0.1744, C=0.1717, D=0.1763, E=0.1752, F=0.1739
- R²: A=0.384, B=0.378, C=0.388, D=0.371, E=0.375, F=0.380
- Params: A=4012, B=10012, C=11492, D=14788, E=1612, F=14788
- Route edge F1 D: 0.0762513808424976, F: 0.05061965698912998
- Preregistered PASS: MSE_D < min_base*0.95 AND F1>=0.30
- Outcome: **UNRESOLVED**
- Notes: True transform low-rank (3-dim diagonal dominant). Hard top-K + STE. Baselines E (recurrent dense), F (gated random graph).

## Run 20261006_151416
- Date: 2026-10-06T15:14:18.780853
- Experiment: TNR v3 – kế hoạch run tiếp theo (capacity↑ R² target, compositional transform, soft+temp, traj metric, E balanced)
- Config: HIDDEN=48, N_ITER=5, TOP_K=3, λ=0.001, EPOCHS=30, T=2.0→0.5
- Results MSE: A=0.1696, B=0.1603, C=0.1653, D=0.1679, E=0.1614, F=0.1693
- R²: A=0.852, B=0.860, C=0.855, D=0.853, E=0.859, F=0.852
- Params: A=15124, B=38644, C=41976, D=48232, E=10420, F=48232
- Route edge F1 D: 0.06436531949071501, traj_node_recall D: 0.4491666666666667
- Route edge F1 F: 0.09463829171488221, traj F: 0.57625
- Preregistered PASS: MSE_D < min_base*0.95 AND F1>=0.30
- Outcome: **UNRESOLVED**
- Notes: Compositional residual true transform (unique dims per module). Soft gates + temperature annealing + soft top-K. Trajectory node recall added.

## Run 20261006_154757
- Date: 2026-10-06T15:48:00.277131
- Experiment: Phương án A – Hierarchical Topological Stacking
- Variants: SameTopo (ring lặp lại) vs DiffTopo (sparse→ring→complete)
- Prior: nhẹ (temperature + soft top-K), KHÔNG teacher forcing / aux path loss
- Results MSE: {'FlatMLP': 0.1617, 'HierResidual': 0.1585, 'SameTopo_static': 0.1624, 'SameTopo_gated': 0.1675, 'DiffTopo_static': 0.1603, 'DiffTopo_gated': 0.1645}
- R²: {'FlatMLP': 0.855, 'HierResidual': 0.858, 'SameTopo_static': 0.854, 'SameTopo_gated': 0.85, 'DiffTopo_static': 0.856, 'DiffTopo_gated': 0.852}
- Params: {'FlatMLP': 9044, 'HierResidual': 40204, 'SameTopo_static': 47972, 'SameTopo_gated': 57332, 'DiffTopo_static': 47972, 'DiffTopo_gated': 57332}
- Best gated: DiffTopo_gated = 0.1645 | Best baseline = 0.1585
- Outcome: **UNRESOLVED**
- Notes: Message-passing song song trong lớp. Inter-layer residual. So sánh SameTopo vs DiffTopo.

## Run 20261006_155655
- Date: 2026-10-06T15:56:58.953298
- Experiment: Hierarchical DiffTopo + prior P0/P1/P2
- P0: light (temp+topK) | P1: +entropy | P2: +curriculum soft bias
- Results MSE: HierRes=0.1580, Static=0.1602, P0=0.1629, P1=0.1621, P2=0.1645
- Entropy: P0=0.5439305901527405, P1=0.00567987933754921, P2=0.1736096888780594
- Outcome: **UNRESOLVED**
- Notes: Tăng prior có kiểm soát trên DiffTopo hierarchical. Không full teacher forcing.

## Run 20261006_160550
- Date: 2026-10-06T16:05:52.309047
- Experiment: Explicit Sequential Trajectory (gọi module tuần tự) trên DiffTopo hierarchical
- Variants: SeqSoft (softmax) vs SeqHard (Gumbel-softmax)
- Results MSE: HierRes=0.1580, SeqSoft=0.1628, SeqHard=0.1586
- Traj recovery: Soft=0.41611111111111104, Hard=0.3872222222222222
- Outcome: **UNRESOLVED**
- Notes: State truyền tuần tự, controller chọn module mỗi bước. Topology mask DiffTopo. Không parallel message-passing.

## Run 20261006_161628
- Date: 2026-10-06T16:16:30.921348
- Experiment: Sequential Trajectory on Variable Long-Dependency + OOD Composition
- Train paths length 3–4 | OOD paths length 5–6 + new compositions
- Results OOD MSE: HierRes=0.1775, SeqSoft=0.1840, SeqHard=0.1809
- Traj seq recovery OOD: Soft=0.26825000000000004, Hard=0.24883333333333335
- ID MSE: HierRes=0.0951, Soft=0.0985, Hard=0.0967
- Outcome: **UNRESOLVED**
- Notes: Metric chính OOD MSE + LCS-based traj sequence recovery. SeqHard ưu tiên.
