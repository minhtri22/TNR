
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

## Run 20261006_162524
- Date: 2026-10-06T16:25:26.365179
- Experiment: SeqHard + Curriculum path-length + Aux sequence loss
- Results OOD MSE: HierRes=0.1745, SeqHard_plain=0.1793, SeqHard_CurrAux=0.3093
- Traj seq recovery OOD: plain=0.2813333333333333, CurrAux=0.30750000000000005
- Outcome: **FAIL**
- Notes: Curriculum warmup 10 ep short paths; AUX_LAMBDA=0.02.

## Run 20261006_164230
- Date: 2026-10-06T16:42:32.562552
- Experiment: Sequential + sliding KV table per node + attention styles (scaled_dot / additive / gated)
- Results OOD MSE: HierRes=0.1764, SeqNoMem=0.1743, scaled_dot=0.1721, additive=0.1709, gated=0.1736
- Traj recovery OOD: NoMem=0.24314285714285713, best_KV(additive)=0.31438095238095237
- Outcome: **UNRESOLVED**
- Notes: KV trượt = overwrite slot module được chọn bằng state mới. Controller attend KV rồi chọn module tiếp.

## Run 20261006_165236
- Date: 2026-10-06T16:52:38.280892
- Experiment: KV_additive param-balanced + mild seq aux + multi-seed [42, 123, 7]
- Params: HierRes≈40204, KV≈55915
- OOD MSE mean: HierRes=0.1866±0.0025, KV=0.1791±0.0055, KVAux=0.1791±0.0055
- Traj OOD mean: KV=0.303, KVAux=0.303
- Outcome: **UNRESOLVED**
- Notes: Cân bằng param (HIDDEN=40, KV_DIM=24). Aux λ=0.015 soft path consistency.

## Run 20261006_165820
- Date: 2026-10-06T16:58:22.482377
- Experiment: So sánh KV_additive vs KV_causal (sliding KV, additive attention)
- Seeds: [42, 7]
- OOD MSE mean: HierRes=0.1808, Additive=0.1753, Causal=0.1731
- Traj mean: Additive=0.306, Causal=0.347
- Winner OOD: **Causal**
- Notes: Causal = chỉ attend slots đã write. Additive = attend full KV table.

## Run 20261007_kv_causal_confirm_partial
- Date: 2026-10-07
- Experiment: KV_causal củng cố – 4/5 seeds + param attempt
- Params: HierRes=32728, KV=45115 (ratio=1.38)
- OOD MSE (4 seeds): HierRes=0.1840±0.004, KV_causal=0.1770±0.005
- Traj recovery: 0.256±0.026
- Wins: 4/4 seeds
- Outcome: **UNRESOLVED**
- Notes: Tín hiệu nhất quán nhưng gap và traj chưa đạt ngưỡng preregister. Seed 2024 timeout.

## Run 20261006_173757
- Date: 2026-10-06T17:37:59.638626
- Experiment: KV_causal parity ratio~0.80 + đủ 5 seeds
- Params: HR=44230, KV=35467
- OOD: HierRes=0.1844±0.0051, KV=0.1817±0.0033
- Traj: 0.274±0.035 | Wins: 4/5
- Outcome: **UNRESOLVED**

## Run 20261007_004156
- Date: 2026-10-07T00:41:56.797157
- Experiment: Xen kẽ KV_causal với 1 lớp Full all-to-all (Causal–Full–Causal)
- Seeds: [42, 7, 123]
- OOD: HierRes=0.1818, Pure=0.1791, Inter=0.1760
- Traj: Pure=0.250, Inter=0.266
- Outcome: **PASS** (best=Inter)
- Notes: Layer giữa = mọi module nối mọi module (parallel message). Hai lớp ngoài = KV_causal.

## Run 20261007_004313
- Date: 2026-10-07T00:43:14.607369
- Experiment: Xen kẽ KV_causal với 1 lớp Full all-to-all (Causal–Full–Causal)
- Seeds: [42, 7, 123]
- OOD: HierRes=0.1818, Pure=0.1791, Inter=0.1760
- Traj: Pure=0.250, Inter=0.266
- Outcome: **PASS** (best=Inter)
- Notes: Layer giữa = mọi module nối mọi module (parallel message). Hai lớp ngoài = KV_causal.

## Run 20261008_041827
- Date: 2026-10-08T04:18:29.276534
- Experiment: Spark-X2.5 inspired Hybrid (CCFC=3C+1F) vs R21(CFC) vs R11(CFCF)
- Audit prior PASS: razor-thin, param +36%, only 3 seeds → treated as fragile
- OOD mean: HR=0.1880, Spark=0.1793, R21=0.1804, R11=0.1813
- Traj: Spark=0.258, R21=0.232, R11=0.215
- Best=Spark, ratio=1.53, wins=3/3
- Outcome: **UNRESOLVED**

## Run 20261008_050749
- Date: 2026-10-08T05:07:50.703782
- Experiment: **Phương án B** — Multi-hop routing-critical task v1
- Task: operator rời rạc (không residual mượt), train path len 3, OOD len 5 + composition mới
- Backbone: PureC (CCC), Spark (CCFC), HierResidual
- OOD: HR=0.5035, PureC=0.5006, Spark=0.5069
- Traj: PureC=0.457, Spark=0.427
- Best=PureC, wins=2/3
- Outcome: **UNRESOLVED**
- Notes: Preregister chặt (0.95× + traj≥0.40). Bắt đầu chuỗi PASS/FAIL để rút kiến trúc.

## Run 20261008_050920
- Date: 2026-10-08T05:09:21.197753
- Experiment: **Phương án B** — Multi-hop routing-critical task v1
- Task: operator rời rạc (không residual mượt), train path len 3, OOD len 5 + composition mới
- Backbone: PureC (CCC), Spark (CCFC), HierResidual
- OOD: HR=0.5035, PureC=0.5006, Spark=0.5069
- Traj: PureC=0.457, Spark=0.427
- Best=PureC, wins=2/3
- Outcome: **UNRESOLVED**
- Notes: Preregister chặt (0.95× + traj≥0.40). Bắt đầu chuỗi PASS/FAIL để rút kiến trúc.

## Run 20261008_multihop_v2_partial
- Date: 2026-10-08
- Experiment: Phương án B v2 — tune difficulty (path ngắn, operator êm)
- Partial 2/3 seeds: HR OOD R²≈0.76 (target was 0.3–0.5) — **too easy**
- OOD: HR≈0.130, PureC≈0.134, Spark≈0.132
- Traj: PureC≈0.41, Spark≈0.49
- Outcome: **UNRESOLVED** + difficulty overshoot
- Notes: Need v2.5 middle ground between v1 (R²~0.09) and v2 (R²~0.76)

## Run 20261008_053654
- Date: 2026-10-08T05:36:56.043554
- Experiment: **Phương án B v2.5** — Multi-hop middle difficulty
- HR OOD R²=0.565 (target 0.3–0.5)
- OOD: HR=0.1286, PureC=0.1255, Spark=0.1285
- Traj: PureC=0.404, Spark=0.453
- Best=PureC, wins=3/3
- Outcome: **UNRESOLVED**

## Run 20261008_spiral_partial
- Date: 2026-10-08
- Experiment: C-C-Spiral-C-C-F on multi-hop v2.5
- Partial 2/3 seeds: OOD HR≈0.128, PureC≈0.127, Spiral≈0.128
- Traj: PureC≈0.39, Spiral≈0.53 (higher)
- Params Spiral≈94k vs HR 40k
- Outcome: **UNRESOLVED** (OOD no gain; traj up; heavy params)

## Run 20261008_061256
- Date: 2026-10-08T06:12:58.689042
- Experiment: **2C + Spiral + F (CCSF)** trên multi-hop v2.5
- OOD: HR=0.1268, PureC=0.1276, CCSF=0.1284
- Traj: PureC=0.389, CCSF=0.433
- Best=PureC, wins=1/3
- Outcome: **UNRESOLVED**

## Run 20261008_062448
- Date: 2026-10-08T06:24:49.353459
- Experiment: **Winnow solo** (sàng thóc) trên multi-hop v2.5 — chưa lắp Causal/Spiral
- Res OOD=0.1256; Best winnow=Res_WinnowLearn_mid OOD=0.1259 sparsity=0.01
- Outcome: **UNRESOLVED**
- Notes: Mag=topk |h|; Learned=sigmoid gate. input vs mid position.

## Run 20261008_063013
- Date: 2026-10-08T06:30:13.863785
- Experiment: **Split heavy/light + gated recombine** (không loại trấu) — solo
- OOD: Res=0.1251, MagGate=0.1263, LearnGate=0.1228
- Outcome: **UNRESOLVED**

## Run 20261008_063626
- Date: 2026-10-08T06:36:27.611948
- Experiment: **Spiral solo** (chưa gắn Causal) — giả thuyết tách ý nghĩa hút lẫn nhau
- OOD: Res=0.1266, Best=SpiralOnly_1=0.1250, |theta|=0.142
- Outcome: **UNRESOLVED**

## Run 20261008_064207
- Date: 2026-10-08T06:42:07.945920
- Experiment: **Nén khuôn solo** (bottleneck mold) — chưa lắp TNR
- OOD: Res=0.1273, Best=MoldOnly_narrow=0.1240
- Outcome: **UNRESOLVED**

## Run 20261008_064714
- Date: 2026-10-08T06:47:15.109523
- Experiment: **Mold hẹp + Spiral combo solo** (Mold→S, S→Mold, Parallel)
- OOD: Res=0.1249, Best=Res=0.1249
- Outcome: **UNRESOLVED**

## Run 20261008_065124
- Date: 2026-10-08T06:51:27.710006
- Experiment: Multi-seed n=100 Res vs Mold vs Mold→Spiral (light train)
- Res=0.1251±0.0045
- Mold=0.1249±0.0045 wins=53/100
- MoldSpiral=0.1253±0.0043 wins=40/100
- Note: requested 1000; ran 100 (feasible). Statistical picture stabilizes.

## Run 20261008_071006
- Date: 2026-10-08T07:10:08.346375
- Experiment: **PureC difficulty ladder (B)** — chỉ HR vs PureC, 3 mức EASY/MID/HARD
- Không add-in. Preregister PASS: HR_R2∈[0.25,0.55] ∧ OOD<0.95×HR ∧ traj≥0.40 ∧ wins≥2/3
- Results in metrics.json / console summary

## Run 20261008_071123
- Date: 2026-10-08T07:11:24.660862
- Experiment: **PureC difficulty ladder (B)** — chỉ HR vs PureC, 3 mức EASY/MID/HARD
- Không add-in. Preregister PASS: HR_R2∈[0.25,0.55] ∧ OOD<0.95×HR ∧ traj≥0.40 ∧ wins≥2/3
- Results in metrics.json / console summary

## Run 20261008_084801
- Date: 2026-10-08T08:48:04.020854
- Experiment: **Task mới Lookup-Chain** (hard transform select, weak endpoint hint)
- Backbone: chỉ HR vs PureC
- HR OOD R²=-0.800 (target band 0.25–0.55: NO)
- OOD: HR=0.8023, PureC=0.7990, traj=0.497, wins=2/3
- Outcome: **UNRESOLVED**

## Run 20261008_layout_partial
- Date: 2026-10-08
- Experiment: **A-core PureC ± layout prior** (2/3 seeds completed)
- Task: Lookup-Chain L2→L3 soft
- HR OOD≈0.488 R2 negative
- PureC no-prior OOD≈0.391 traj≈0.39 (ratio ~0.80 vs HR)
- PureC+layout OOD≈0.378 traj≈0.45 (ratio ~0.77)
- Both PureC beat HR by ~20%+ OOD — strongest core signal to date
- Outcome: **PASS-leaning UNRESOLVED** (need seed 123; traj prior helps)
- Notes: Core-only; no add-ins. Layout prior raises traj toward 0.45.

## Run 20261008_121151
- Date: 2026-10-08T12:11:52.732492
- Experiment: **A-core PureC + layout prior** (bỏ hết add-in)
- Task: Lookup-Chain L2→L3
- HR OOD=0.5123 R2=-0.834
- PureC no-prior OOD=0.3869 traj=0.468
- PureC+layout OOD=0.4154 traj=0.418
- Best=PC0, wins=1/3
- Outcome: **UNRESOLVED**

## Run 20261008_layout_3seed_complete
- Date: 2026-10-08
- Experiment: A-core PureC ± layout — **3 seeds full train complete**
- HR OOD=0.496±0.017
- PureC no-prior OOD=0.389±0.009 traj=0.415 wins=3/3 ratio=0.79
- PureC+layout OOD=0.390±0.038 traj=0.442 wins=3/3 ratio=0.79
- Outcome: **UNRESOLVED** (OOD PASS-level ~21% better; traj 0.44 borderline 0.45)
- Note: light 10-seed (under-train) flipped — HR stronger when PureC undertrained; full budget needed for routing signal

## Run 20261008_layout_tighten_partial
- Date: 2026-10-08
- Experiment: Siết layout prior λ=0.25 / align map (partial seed 42)
- Seed 42: HR=0.504 | PC0 OOD=0.402 traj=0.463 | L15 OOD=0.336 traj=0.488 | L25 OOD=0.379 traj=0.358 | L25A OOD=0.532 traj=0.367
- Finding: **λ=0.25 and align map HURT** traj and OOD vs λ=0.15
- Siết prior quá mạnh = FAIL có kiểm soát (credit assignment over-constraint)
- Best still L15 (λ=0.15 spread) — traj 0.49 on this seed
- Outcome partial: **FAIL for tighter prior**; keep λ=0.15

## Run 20261008_5seed_repro_partial4
- Date: 2026-10-08
- Experiment: 5-seed reproducibility (4/5 complete; seed 2024 pending)
- HR OOD≈0.515
- PC0 OOD≈0.413 traj≈0.38 wins=4/4 ratio≈0.80
- PC1 (λ=0.15) OOD≈0.430 traj≈0.47 wins=3/4 ratio≈0.84
- PC1: traj **PASS threshold** (0.47≥0.45); OOD <0.95×HR; wins 3/4 (need 4/5 for formal PASS)
- Seed 99 outlier: PC1 OOD worse than HR
- Outcome provisional: **UNRESOLVED** (strong; formal PASS blocked by 1/4 OOD miss + missing 5th seed)

## Run 20261008_5seed_repro_partial4
- Experiment: 5-seed reproducibility (4/5 complete)
- HR OOD=0.515
- PC0 OOD=0.413 traj=0.38 wins=4/4 ratio=0.80
- PC1 λ=0.15 OOD=0.430 traj=0.47 wins=3/4 ratio=0.84
- traj PC1 exceeds 0.45; formal PASS needs wins 4/5
- Outcome: UNRESOLVED (strong)


## Run 20261008_141050
- Date: 2026-10-08T14:10:52.017964
- Experiment: **5-seed full budget reproducibility** PureC λ=0.15 vs λ=0 vs HR
- HR=0.5053±0.0138
- PC0=0.5166 traj=0.515 wins=1/5
- PC1=0.3713 traj=0.340 wins=2/5
- Outcome: **UNRESOLVED**

## Run 20261008_5seed_COMPLETE
- Date: 2026-10-08
- Experiment: **5-seed full budget COMPLETE** (42,7,123,99,2024)
- HR OOD=0.510
- PC0 OOD=0.435 traj=0.40 wins=4/5
- **PC1 λ=0.15 OOD=0.413 traj=0.457 wins=4/5 ratio=0.81**
- Preregister: OOD<0.95×HR ✓ | traj≥0.45 ✓ | wins≥4/5 ✓
- Outcome: **PASS**
- Note: seed 99 still outlier on OOD for PC1; overall criteria met. Extra seed 11 PC1 wins but traj low.

## Run 20261008_142013
- Date: 2026-10-08T14:20:14.163972
- Experiment: **Typed modules** vs PureC homogeneous (λ=0.15) — điểm gãy specialization
- HOM OOD=0.4092 traj=0.431
- TYP OOD=0.4158 traj=0.417 wins_vs_hom=1/3
- Outcome: **UNRESOLVED**

## Run 20261009_064434
- Date: 2026-10-09T06:44:38.275687
- Experiment: **Curriculum MIX** phase2 (L2+L3) + λ=0.20 vs BASE / REP
- BASE OOD=0.3637 traj=0.433
- REP  OOD=0.4089 traj=0.390
- MIX  OOD=0.3246 traj=0.374 wins=2/3
- Outcome: **UNRESOLVED**

## Run 20261009_curriculum_mix
- Date: 2026-10-09
- Experiment: Curriculum MIX L2+L3 phase2 + λ=0.20
- Combined seeds 42+7+123: MIX OOD≈0.33 vs BASE≈0.38 (ratio~0.86, wins 3/3)
- MIX traj≈0.38 (still <0.45) — OOD gain kept, traj not restored
- Outcome: **UNRESOLVED**

## Run 20261009_070548
- Date: 2026-10-09T07:05:51.842033
- Experiment: **n_mod=6 (=N_KEYS)** vs n_mod=4, λ=0.15, 5 seeds
- N4 OOD=0.4162 traj=0.413
- N6 OOD=0.3364 traj=0.464
- traj≥0.45: True; OOD≤1.05×N4: True
- Outcome: **PASS**

## Run 20261009_nmod6_5seed_COMPLETE
- Date: 2026-10-09
- Experiment: **n_mod=6 (=N_KEYS)** vs n_mod=4, λ=0.15, 5 seeds full
- N4 OOD=0.442 traj=0.418
- N6 OOD=0.377 traj=0.468
- traj≥0.45 ✓ | OOD≤1.05×N4 ✓ (ratio 0.85) | vs REF PASS also OK
- Outcome: **PASS**
- Collision fix helps traj; OOD không xấu (thậm chí tốt hơn mean)

## Run 20261009_A2_lambda0_partial4
- Date: 2026-10-09
- Experiment: **A2 λ=0 vs λ=0.15**, n_mod=6 (4/5 seeds; 2024 pending)
- L15 OOD≈0.391 traj≈0.441
- L0  OOD≈0.386 traj≈0.531
- traj≥0.45 ✓ | OOD≤1.05×L15 ✓
- Outcome provisional: **PASS-leaning** (λ=0 không phá; traj thậm chí cao hơn)
- Next: hoàn tất seed 2024; nếu giữ → B không bắt buộc; có thể A3 rồi C
