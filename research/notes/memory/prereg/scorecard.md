## Scorecard

Every bar in this registration and its amendments, with the number that
decided it. Cells are (n, k); 20 brains unless stated.

| Bar | Registered | Verdict | Deciding number |
|-----|-----------|---------|-----------------|
| R1 capacity | REF >= 4x CTL at (4000, 60) | PASS | 1961 vs 83, 23.5x |
| R2 ratio law at fixed k | M*(n)/n constant | FAIL | superseded by R6/R7: the law is in n/k |
| R3 orthogonalization | pairwise overlap below chance at M* | FAIL as stated, PASS restated | overlap at chance past fill 1.0; distinct 1.000 |
| R4 masked readout | net readout at chance | PASS | net M* = 8 |
| R5 rounds | T = 16 raises the ceiling | FAIL (inverted) | 203 at T = 16 vs 1961 at T = 8 (A7 correction) |
| R6 control law in n/k | matched cells within 25% | PASS | 64/83, 89/83, 263/307, 11.3/11.3 |
| R7 refracted law in n/k | matched cells within 25% | PASS at 33 and 67 | 1589/1961, 2230/1961, 383/431; at 133 the pair disagrees by 0.68 (one cell out of regime) |
| N1 numpy engine | REF >= 3x CTL | PASS | >= 512 vs 34 (5 brains) |
| N2 numpy distinct | distinct 1.000 at M* | PASS | 1.000 |
| Q1 quadratic | M*/(n/k)^2 in [0.35, 0.50] at both n/k = 133 cells | PASS at (8000, 60), FAIL LOW at (4000, 30) | 0.395; 0.27 (out of regime) |
| Q2 multiplier | reported | -- | 23x and 18x at n/k = 133; the control is quadratic too from n/k = 67 |
| Q3 resolution | bracket <= 1.5x | PASS | [6144, 8192); [4096, 6144) |
| Q4 distinct | distinct >= 0.99 below M* | PASS | 1.000 |
| G1 gate not worse | gated M* >= 1536 | PASS | 2645 |
| G2 gate better | gated M* > 2048 | PASS | 2645 [2560, 2816) |
| G3 rounds fall with load | fewer rounds at high load | FAIL | a U: 5.8 rounds at M ~ 400, 8 at M >= 2048 |
| G4 gated control | reported | -- | no memory forms: rank-1 0.32 at M = 8 |
| G5 out-of-regime cell, T_max 16 | converges and reaches the pair | FAIL | converges (0.956) and M* falls 13x to 362 |
| G6 gain constant in n/k | ratio in [1.15, 1.55] at (8000, 60) | PASS | 1.24 (8666 vs 6995) |
| S8 strength a lever | some strength >= 2275 | FAIL | 1986, 1919, 1961, 1993 at 0.3 to 0.6 beta (A7 correction) |
| S9 convergence | reported | -- | all converge; transition in (0.6, 0.7] beta |
| RP-1 exact reproduction (A8) | surviving legacy values equal | PASS | 3,615 + 3,050 per-seed values equal |
| RP-2 lost ceilings (A8) | inside the reported brackets | PASS | all seven, within 0.1% of the reports |
| RP-3 law as cited (A8) | in-regime M*/(n/k)^2 in [0.35, 0.50] | FAIL | 0.345 and 0.502: the quoted range was rounded inward |
| RP-4 live contrast (A8) | every brain >= 0.50 at M_s, six cells | PASS | minimum 0.938 |
| PE-V instrument (A9) | recall agreement 0.03; M*_real within 10% of A8 | PASS | gap 0.000; 0.979-1.040 |
| PE-0 calibration (A9) | random pair-sharing excess in [0.9, 1.1] | PASS | 0.998-1.001 |
| PE-1 substrate is Willshaw (A9) | M*_random in [0.5, 1.0] (n/k)^2, potentiated [0.40, 0.60] | PASS | 0.69-0.92; 0.49-0.60 |
| PE-2 constant efficiency (A9) | eta in [0.35, 0.70], max/min <= 1.5 | PASS | 0.47-0.56; 1.19 |
| PE-3 pair reuse is the gap (A9) | excess >= 1.5, failure pair load within [0.75, 1.33] of random's | FAIL | 1.43 at (4000, 120) |
| PE-4 transients help (A9) | M*_real > M*_clean | PASS | 1.17-1.47x |
| PE-5 balanced headroom (A9) | M*_balanced >= 1.5 M*_random | FAIL | 1.36 at (8000, 60) |
| PE-S weight sensitivity (A9) | c +- 2 within 30% | FAIL | +66 to +89% at k = 60 |
| PE-R gate replayed (A9) | within 10% of 2645 and 8666 | PASS | 0.980, 1.048 |
| PE-6 gate lowers pair reuse (A9) | eta_gated > eta, lower sharing, failure load within [0.75, 1.33] | FAIL | 1.39 at (8000, 60) |
| WS-V instrument (A10) | own-c ceiling replays A9 within 10% at every cell | FAIL | 16 vs 5043 and 20741 at k = 30 |
| WS-1 .. WS-5 (A10) | -- | VOID | adaptive search assumed one crossing |
| XV instrument (A11) | own-c upper edge within 10% of A9, all cells | PASS | 0.978-1.012 |
| X1 completion needs a strong write (A11) | c = 32 >= 1.5x own c | PASS | 1.9x to 37x; own c 0 at two cells |
| X2 binary completes what it identifies (A11) | ratio >= 0.85 | PASS | 0.88-0.99 |
| X3 separable law at binary write (A11) | within-k 15%, k = 120 >= 1.25x k = 60 | PASS | 0.560-0.563 vs 0.798; 1.42x |
| X4 n/k law belongs to the operating write (A11) | own-c pairs in [0.8, 1.25], c = 32 pairs <= 0.8 | PASS | 1.11, 0.89; 0.68, 0.69 |
| X5 weak writes have a load window (A11) | c = 3 lower edge, capacity above own c | PASS | 213-801 lower; 1.6-1.8x |
| X6 smallest completing write falls with k p (A11) | strictly decreasing in k per n/k level | PASS | 11 / 6 / 4 |
| TV instrument (A12) | beta = 0.1 rank-1 within 10% of A7/A8 | PASS | 0.979-1.040 |
| T1 interior optimum (A12) | completion peaks inside the beta grid | FAIL | (2000, 60): merged recall at the top |
| T2 beta* falls with k p (A12) | strictly decreasing per n/k level | FAIL | 0.209 at (8000, 240), merged recall |
| T3 gamma* within 25% (A12) | all five cells | FAIL | undefined; 1.47 |
| T4 wider is better at the optimum (A12) | best completion / (n/k)^2 rises with k p | PASS | 0.66 / 1.12 / 2.79; 0.34 / 0.94 |
| UV instrument (A13) | beta = 0.1 rank-1 within 10% of A7/A8, new brains | PASS | 0.968-1.058 |
| U1 interior optimum (A13) | distinct completion peaks inside the grid, 7 cells | PASS | all seven |
| U2 the law (A13) | gamma* within 15% of 0.29, 7 cells incl. 2 unseen | PASS | 0.257-0.303 |
| U3 the exponent (A13) | 1/2 collapses best, CV < 0.10 | PASS | 0.052 (raw beta 0.213) |
| U4 wider is better (A13) | best distinct completion / (n/k)^2 rises with k p | PASS | 0.41 to 2.46; 0.34, 0.94 |
| WV instrument (A14) | A13 capacities within 15% on new brains | PASS | 0.970-1.077 |
| W1 recall multiplier (A14) | refracted >= 10x control, each at its best beta | PASS | 14.0-57.3 at ten cells |
| W2 synapse-count form (A14) | C within 15% and tighter than (n/k)^2, per block | FAIL | block B: CV 0.10 vs 0.07 |
| W3 connectivity only as k p (A14) | capacity within 15% and same best beta | FAIL | capacity within 8%; best beta differs |
| W4 learning-rate law at sparse p (A14) | best beta interior at p < 0.5 | FAIL | two at the sweep's top |
| DV instrument (A15) | T = 8 replays A14 within 15% | PASS | 486, 1575 |
| D1 old effect (A15) | beta 0.1: T = 16 below T = 8 | PASS | 86 vs 287; 207 vs 977 |
| D2 rescaled write transfers (A15) | law point within 15% across T | FAIL | T = 12, 16 far below |
| D3 best rate as 1/T (A15) | T ln(1+beta*) within 20% | FAIL | T = 16 optimum at the grid's bottom |
| D4 depth does not matter at the optimum (A15) | best capacity within 15% across T | PASS | within 9% at both cells |
| SV instrument (A16) | A14 capacities within 15% on new brains | PASS | 0.974-1.018 |
| S1 interior optimum (A16) | all six cells | PASS | all six |
| S2 noise law (A16) | gamma_noise within 15%, tighter than gamma_plain | PASS | CV 0.081 vs 0.258 |
| S3 dense side (A16) | beta*(p .75) / beta*(p .5) <= 0.85 | PASS | 0.707 (predicted 0.71) |
| S4 capacity p-invariant (A16) | k p = 30 bests within 15% | PASS | 436-497 |
| TV instrument (A17) | (4000, 60, 0.5) within 15% of 1616 | PASS | 1589 (0.983) |
| T1 threshold fraction above floor (A17) | beta*/theta within 15% of 0.20 at k p = 30, 40, 80 | PASS | 0.180, 0.189, 0.180 |
| T2 threshold fraction below floor (A17) | >= 1 resolved; every resolved within 25% of 0.20 | FAIL | p 0.5: 0.19-0.22; p 0.05: 0.21, 0.27, 0.30; PNAS cell never completes |
| OV instrument (A18) | (4000, 60, 0.5) within 15% of 1589 | PASS | 1597 |
| O1 universal onset (A18) | onset/theta in [0.13, 0.21] at all ten, CV <= 0.10 | PASS | 0.160-0.180, CV 0.041 |
| O2 dense onset load-assisted (A18) | onset window opens at >= 64 items, p = 0.5 | PASS | 158-1098 |
| O3 sparse onset immediate (A18) | no lower edge at p = 0.05 | FAIL | none / 3 items |
| O4 dense optimum at onset (A18) | best/onset <= 2^(1/4), p = 0.5 | FAIL | 1.00-1.06 at 5 cells; 1.1897 at (8000, 120) |
| O5 sparse optimum above onset (A18) | best/onset >= 1.4, p = 0.05 | PASS | 1.78, 1.59 |
| XV instrument (A19) | A17 capacities within 15% | PASS | 1.01, 1.12 |
| X1 flat below floor (A19) | k = 10-40 within 15% of mean | FAIL | k = 14 at -16% |
| X2 linear in n below floor (A19) | x1.6-2.5 per doubling | FAIL | x2.98, x2.13 |
| X3 falling above floor (A19) | strictly decreasing, k=160 <= 0.8 k=56 | FAIL | rises 56 -> 80; 0.76 |
| R1 gap grows as fan-in falls (A20) | Spearman(k p, R) <= -0.7 | PASS | -0.883 |
| R2 recall needs fan-in (A20) | rank-1 >= 32 all; distinct 0 at kp 1, >= 32 at kp 8 | PASS | 114-207; 0; 40 |
| R3 divergence large (A20) | R(k 10) >= 3 R(k 160) | FAIL | 1.78x |
| DV instrument (A21) | (4000, 60, 0.5) within 15% of 1597 | PASS | 1608 |
| D1 degree-law prediction (A21) | new cells within 25% of 0.0135 d^1.51 | FAIL | 7 of 8; d = 6000 at 1.257 |
| D2 in-degree alone (A21) | equal-d cells within 20% of mean, n/k 2-4x | PASS | max 11% |
| D3 degree-law exponent (A21) | slope in [1.35, 1.65] | FAIL | 1.669 |
| A22 recognition at own optimum | -- | VOID | chance rank-1 at M = 2 armed the stop rule |
| QV instrument (A23) | recall at (4000, 40, 0.5) within 15% of 1473 | PASS | 1339 |
| Q1 recognition outruns recall (A23) | ratio >= 2 at every cell | PASS | 15.0-86.3 |
| Q2 gap widens as fan-in falls (A23) | Spearman <= -0.9 and k=10/k=160 >= 5 | FAIL | -0.90; 4.0x |
| Q3 recognition wants a weaker write (A23) | recognition rate <= half recall's | PASS | 1/40-1/6 |
| CV instrument (A25) | round write within [0.75, 1.10] of A18 | PASS | 0.89, 0.93, 0.98 |
| W1 deferral abolishes the memory (A25) | deferred capacity 0 at every rate, every cell | PASS | 0 |
| W2 burst write stores nothing (A25) | burst capacity 0 at every rate, every cell | PASS | 0 |
| W3 burst gating costs capacity (A25) | online_burst <= half round's | PASS | 0 (abolished) |
| W4 deferred write stores the trajectory (A25) | next >= 0.4, >= 5x same, own <= 0.1; round own >= 0.5 | PASS | next 0.64-0.79; own 0.002-0.022; round own 0.80-0.98 |
| S1 deferred write is a sequence memory (A26) | peak replay >= 0.9, own <= 0.1 every rate | PASS | 1.00; own <= 0.042 |
| S2 adaptation switches the memory type (A26) | online s=1.5: own <= 0.1, peak >= 0.7; s=0.5: own >= 0.5 | PASS | own <= 0.052, peak 1.00; control 0.71-0.85 |
| S3 sequence capacity follows n/k (A26) | same n/k within 25%, same d >= 2x | FAIL | 0.48; 1.68 |
| S3d sequence capacity follows in-degree (A26) | same d within 25%, same n/k <= 0.5 | FAIL | 1.68; 0.48 |
| L1 Hebbian sequence limit is Willshaw's (A27) | L_H / (p (n/k)^2) in [0.06, 0.12] | PASS | 0.063-0.106 |
| L2 sequence limit follows n/k (A27) | equal-n/k pairs within 30% | FAIL | 22%; 67% |
| L3 Hebbian failure is a cliff (A27) | ensemble >= 0.9 at L_H/sqrt2, <= 0.2 at sqrt2 L_H | FAIL | 3 cells; all-or-none per brain (post hoc) |
| L4 refraction extends the limit (A27) | L_R >= 2 L_H every cell | FAIL | 2.2-31x at n/k <= 67; 0.33, 0.57 at n/k = 133 |
| D1 the sequence tiles the area (A28) | fresh >= 0.95 to 0.9 n/k, <= 0.10 from 1.1 n/k | PASS | every cell |
| D2 replay breaks on the deadline (A28) | >= 80% of breaks within 5% of j n/k | PASS | 49 of 51 (96%) |
| D3 recovering bias removes it (A28) | reset >= 0.9 every cell, +0.3 at n/k = 133 | PASS | 1.00 everywhere; 0.26, 0.37 refracted |
| B1 recovery beats both ends (A29) | best >= 2x max(Hebbian, cumulative) every cell | FAIL | 1.27 at (2000, 60); 4.2-8.0 elsewhere |
| B2 the optimum is interior (A29) | best tau neither 0 nor 512 | PASS | 32, 64, 64, 64 |
| B3 upper edge scales with n/k (A29) | edge in [0.5, 4] n/k | FAIL | 15 n/k at (2000, 60); 1.9-3.8 elsewhere |
| B4 best limit grows as (n/k)^2 (A29) | best/(n/k)^2 spread <= 1.5 | FAIL | 0.14-0.61; follows the in-degree (post hoc) |
| C1 refraction codes shared elements apart (A30) | overlap <= 0.05, >= 18/20 right, gap 0 | PASS | <= 0.016; 20/20 |
| C2 without refraction the context is lost (A30) | m = 16: overlap >= 0.5, <= 10/20 right | PASS | 0.83-0.87; 0/20 |
| C3 separation is largely recency (A30) | overlap >= 0.15 after 600, every m | FAIL | 0.09-0.11 at m = 1; 0.67-0.70 at m = 16 |
| R1 forward write is forward only (A31) | r = 0: forward >= 0.9, backward <= 0.1 | PASS | 1.00; 0.00 |
| R2 one reverse count too weak (A31) | r = 1: backward <= 0.5 | PASS | 0.00-0.01 |
| R3 two counts + LRI both ways (A31) | r = 2, 3: all four reads >= 0.9 | FAIL | backward 1.00; forward 0.02-0.29 |
| R4 no LRI no direction (A31) | r = 2: forward_masked <= 0.5 | PASS | 0.01 |
| Q1 balanced chain both ways (A32) | 2 + 3: four reads >= 0.9 | PASS | 1.00 everywhere |
| Q2 stronger direction wins (A32) | 2 + 2 forward only; 0 + 2 backward only | PASS | back 0.05-0.33; forward 0.09-0.23 |
| Q3 no LRI no direction, balanced (A32) | 2 + 3 forward without LRI <= 0.5 | PASS | 0.02 |
| H1 sequences of sequences (A33) | (8, 6) link 2: every plan whole >= 0.9 | PASS | 1.00 both cells |
| H2 the links carry it (A33) | link 0: starts <= 0.1 everywhere | PASS | 0.00 |
| H3 plans sharing a run stay apart (A33) | shared-run plans chain, whole >= 0.9 | PASS | 1.00 |
| H4 links give way first, larger pair relieves (A33) | starts < chain - 0.1; larger +0.1 | PASS | 0.78 vs 0.96; 1.00 |
| N1 many sequences share one budget (A34) | budget in [0.6, 1.6] of A29 | PASS | 1.08; 0.81 |
| N2 small noise harmless (A34) | nu 0.05: >= 18/20 full | PASS | 20/20 both cells |
| N3 larger area doubles noise horizon (A34) | median(8000) >= 2 median(4000) at nu 0.1 | PASS | 232 vs 39 |
| N4 cue may be a quarter wrong (A34) | eta 0.25: >= 0.9 | PASS | 1.00 |
| O1 engine equals an independent oracle (A35) | lockstep, counts equal | PASS | ~2,900 rounds, 1 float32 near tie |
| A33 H1-H4 on the corrected divisor (A35) | as registered in A33 | PASS | max change 0.010 |
| U1 uniform noise milder than A34's (A36) | >= 2x at (4000, 60) | PASS | 103.6 vs 41.2 |
| U2 larger area holds under uniform 10% (A36) | >= 18/20 full | PASS | 20/20 |
| U3 A34 reproduces (A36) | within 35% | PASS | 41.2, 207.4 |
| U4 random half cue works (A36) | >= 0.9 | PASS | 1.00 |
| R1 critical load where the law puts it (A37) | rho_50 in [0.109, 0.185], 2 unseen cells | PASS | 0.141, 0.115 |
| R2 reliable below rho 0.09 (A37) | rho_90 >= 0.09 | PASS | 0.132, 0.111 |
| R3 failure is a cliff (A37) | rho_10 / rho_90 <= 1.6 | PASS | 1.15, 1.07 |
| K1 load law's constant holds at n/k 300 (A38) | rho_50 in [0.109, 0.185] | FAIL | 0.081, 0.081 |
| D1 post hoc n/k drift holds (A38) | within x1.2 of 0.104, 0.106 | FAIL | 0.081, 0.081 (nearer) |
| S1 safe rule rho_90 >= 0.09 (A38) | both cells | FAIL | 0.079, 0.078 |
| S2 failure is a cliff (A38) | rho_10 / rho_90 <= 1.6 | PASS | 1.07, 1.08 |
| T1 tau = n/k / 2 raises the cliff (A39) | gain >= 1.2 at n/k 300, > 1 at 200 | PASS | 1.29, 1.18 |
| T2 safe rule returns under the rule (A39) | rho_90 >= 0.09 | PASS | 0.101, 0.107 |
| T3 rho_50 where the probe put it (A39) | [0.100, 0.135] | PASS | 0.105, 0.114 |
| T4 still a cliff (A39) | rho_10 / rho_90 <= 1.6 | PASS | 1.08, 1.11 |
| A1 safe rule holds for any split (A40) | rho_90 >= 0.09, every arm | PASS | 0.111-0.132 |
| A2 short sequences degrade later (A40) | rho_50 ratio >= 1.1 | PASS | 1.13-1.20 |
| A3 short sequences fail one by one (A40) | >= half the brains mixed | PASS | 55-100% |
| P1 tau = n/k peak at n/k 20 (A41) | gain >= 1.3 | PASS | 1.65 |
| P2 tau gain falls with n/k (A41) | Spearman <= -0.8 | PASS | -1.0 |
| P3 safe rule at tau = n/k (A41) | rho_90 >= 0.09, four cells | PASS | 0.093-0.191 |
| P4 no peak at n/k 200 (A41) | gain <= 1.1 | PASS | 0.91 |
| W1 recurring words coded as tokens (A42) | same-word overlap <= 0.3 | PASS | 0.072-0.105 (different 0.000-0.002) |
| W2 moderate reuse is free (A42) | >= 0.98 at <= 5 uses/word | PASS | 1.00 |
| N1 mild noise fits the budget (A42) | >= 0.98, nu <= 0.05, rho <= 0.08 | PASS | >= 0.997 |
| N2 noise and load compound (A42) | drop >= 0.2 at nu 0.1 | PASS | 0.24, 0.60 |
| K1 256 elements predicted from 16/64 (A43) | rho_50 within 5% | PASS | -4.1%, -1.9% |
| K2 1024 elements predicted (A43) | rho_50 within 5% | PASS | -4.7%, -2.2% |
| K3 single cliff predicted (A43) | rho_50 within 5% | FAIL | -9.8%, -4.5% |
| K4 length costs (A43) | rho_50 strictly falls with l | PASS | both cells; ~0.003 per e-fold of l |
| G1 transition repetition costs (A44) | drop >= 0.3 from R~1 to R~10 | PASS | 0.77, 0.95 -> 0.00 |
| G2 recurrence alone is free (A44) | iid at U 20 >= 0.95 | FAIL | 0.771, 0.950 |
| G3 repetition, not recurrence (A44) | R = 5 arms within 0.25 | PASS (vacuous: all 0.000, floor) | uninformative |
| G4 repeated transitions merge tokens (A44) | same-word overlap +0.03 | PASS | +0.059, +0.052 |
| S1 two reuse edges (A45) | both edges at both cells | PASS | U 60: 0.00, 0.09; R 3.7: 0.00, 0.02 |
| S2 separable reuse budget (A45) | interior within 0.15 of f(R) g(U), informative | FAIL | 9/10 within; one +0.156; MAE 0.08, 0.05 |
| H1 read-time adaptation rescues collapse (A46) | habit - masked >= 0.15 | FAIL | +0.237, +0.140 (post hoc: 19/19, 17/17 brains up) |
| H2 no harm to a healthy memory (A46) | habit >= 0.97 | PASS | 1.000, 1.000 |
| H3 repetition failure untouched (A46) | gain <= 0.1 | FAIL | +0.029, +0.109 |
| H4 adaptation has a window (A46) | strong <= masked - 0.1 | FAIL | +0.160, -0.115 |
| Q1 every failing brain helped (A47) | >= 90% up, <= 1 worse | PASS | 25/25, 27/29 up; 0 worse |
| Q2 collapsed brains lifted (A47) | mean gain >= 0.2 | FAIL | 0.234, 0.180 |
| Q3 no harm to a healthy memory (A47) | habit >= 0.97 | PASS | 1.000, 1.000 |
| Q4 rescue mainly the collapse's (A47) | margin over repetition >= 0.2 | FAIL | 0.212, 0.167 |
| M1 the law has one position (A48) | masked midpoint in [0.72, 0.81] | PASS | 0.748, 0.760 |
| M2 adaptation lowers it (A48) | adapted <= masked - 0.06 | PASS | -0.105, -0.110 |
| M3 the hidden margin orders the rescue (A48) | Spearman >= 0.6, >= 8 collapsed | PASS | 0.856 (51), 0.873 (69) |
| M4 the rescue's size is predicted (A48) | collapsed mean within 0.1 | PASS | 0.025, 0.052 |
| W1 no harm (A49) | U 10 separated >= 0.97 | PASS | 1.000, 1.000 |
| W2 the collapse is prevented (A49) | U 50 separated >= 0.85, +0.4 | PASS | 0.995, 0.994 |
| W3 reach (A49) | U 60 separated >= 0.7 | PASS | 0.916, 0.912 |
| W4 no collapsed brains (A49) | <= 2 below 0.2 | PASS | 0, 0 |
| W5 cheap (A49) | <= 3% of writes | PASS | <= 1.9% |
| W6 no capture (A49) | captured <= 0.001 | PASS | 0.000 |
| C1 no harm (A50) | U 10 comparator >= 0.97 | PASS | 1.000, 1.000 |
| C2 rescue (A50) | U 50 comparator >= 0.8, +0.4 | PASS | 0.942, 0.960 |
| C3 no collapsed brains (A50) | <= 2 below 0.2 | PASS | 0, 0 |
| C4 selective (A50) | flags <= 5% (U 50), <= 1% (U 10) | PASS | 2.3%, 1.7%; 0.17%, 0.06% |
| C5 it detects capture (A50) | hit >= 0.6 | PASS | 0.78, 0.84 |
| C6 no capture left (A50) | captured <= 0.001 | PASS | 0.000 |
| S1 repair (A51) | U 50 after 300 >= 0.6, +0.4 | PASS | 0.803, 0.725 |
| S2 no erasure (A51) | 3000 >= 300 - 0.05 | PASS | 0.836, 0.764 |
| S3 no harm (A51) | U 10 >= 0.97, <= 0.1% removed | PASS | 1.000, 0.999; 0 |
| S4 the gate closes (A51) | <= 2% of steps in 1000-3000 | PASS | 0.14%, 0.19% |
| S5 frugal (A51) | <= 10% of counts | PASS | 3.1%, 5.2% |
| S6 no collapsed brains (A51) | <= 2 below 0.2 | PASS | 0, 0 |
| L1 synergy (A52) | both - comparator >= 0.15, - sleep >= 0.3 at U 100 | PASS | +0.40/+0.58, +0.19/+0.36 |
| L2 reach at 80 (A52) | both >= 0.65 | PASS | 0.794, 0.652 |
| L3 reach at 100 (A52) | both >= 0.45 | FAIL | 0.641, 0.368 |
| L4 no collapsed brains (A52) | <= 2 below 0.2 | PASS | 0, 0 |
| L5 a sparing sleep (A52) | <= 2% of counts | PASS | <= 1.35% |
| P1 the edge (A53) | standard at b = 3 <= 0.25 | PASS | 0.001, 0.000 |
| P2 rescue (A53) | both at b = 3 >= 0.75 | PASS | 0.878, 0.923 |
| P3 composition (A53) | both >= comparator, >= sleep + 0.15 | PASS | +0.03/+0.12, +0.38/+0.34 |
| P4 the comparator sees repetition (A53) | flags >= 2% and >= 10x random | PASS | 4.7% vs 0.23%, 4.0% vs 0.06% |
| P5 harmless and sparing (A53) | random >= 0.98, b = 5 >= 0.95, <= 2% removed | PASS | 1.000, 0.998, <= 0.54% |
| P6 no collapsed brains (A53) | <= 2 below 0.2 | PASS | 0, 0 |
| P7 not an immediate merge (A53) | pair overlap <= 0.2 | PASS | 0.113, 0.116 |
| R1 safe (A54) | healthy >= 0.99, <= 0.1% removed | PASS | 1.000, 1.000, 0 |
| R2 reach (A54) | median-gated U 100 >= 0.45 | PASS | 0.712, 0.655 |
| R3 never worse (A54) | median >= max - 0.02 | PASS | +0.012, +0.016 |
| R4 sparing (A54) | <= 2% of counts | PASS | 1.31%, 1.42% |
| R5 no collapsed brains (A54) | <= 2 below 0.2 | PASS | 0, 0 |
| G1 safe (A55) | healthy >= 0.99, <= 0.1% removed | PASS | 1.000, 1.000; <= 0.0007% |
| G2 repair (A55) | set point U 50 >= 0.6 | PASS | 0.802, 0.833 |
| G3 lifecycle reach (A55) | set point U 100 >= 0.45 | PASS | 0.722, 0.717 |
| G4 as good as the reference (A55) | >= median - 0.05 | PASS | +0.02/+0.05, +0.02/+0.05 |
| G5 no collapsed brains (A55) | <= 2 below 0.2 | PASS | 0, 0 |
| G6 frugal (A55) | <= 10% at U 50, <= 5% at U 100 | PASS | 4.4%/2.9%, 3.7%/3.2% |
| G7 the mechanism (A55) | every healthy brain below its set point | FAIL | 19/20, 19/20 |
| CV instrument (A24) | (4000, 60, 0.5) within 15% of 1597 | PASS | 1613 |
| F1 onset sharpens (A24) | sd non-rising (10%), sd(16000) <= 0.6 sd(2000) | FAIL | 0.0031-0.0041 theta, at the grid's resolution at every n |
| F2 onset converges (A24) | means in [0.13, 0.21], 16000/8000 within 10% | PASS | 0.163-0.170; 1.029 |
| F3 read-out slows at onset (A24) | settle(onset) >= 1.5 settle(half-octave up) | FAIL | 0.97-1.39; 42-88% never settle (tie-fragile instrument) |


## Runner migration reproduction (2026-09-10)

The shared runner reproduced the figure control at (4000,60), B, T8,20 seeds,
all 11 recorded load checkpoints. Its [results](../../results/runs/memory.capacity-scaling/migration-capacity-20260910-v3/results.json)
and [comparison receipt](../../results/runs/memory.capacity-scaling/migration-capacity-20260910-v3/comparison.json)
match all 885 compared scalar values, including per-seed metrics and aggregate
ceiling fields. The bracket remains [64,128), too broad to treat interpolated
83.4 as a resolved ceiling. This is migration evidence, not a new adoption.
The original artifact has no run record; reconstruction inputs and their source
are documented in docs/reviews/whole-codebase/VALIDATION.md.


The subsequent [record-consumption replay](../../results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json)
([comparison](../../results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/comparison-v2.json))
also matched all 885 comparisons after arm settings, device and distinctness bars
were changed from implicit globals to required execution inputs from the run
record. The registered defaults and scientific interpretation are unchanged.

