# Reinforcement Learning Project

## Retail Inventory DQN: reproducible custom-simulator study

Revision of the user-supplied `RL_Inventory_Management_FINAL (1).ipynb` and Project 5 brief. Original files remain unchanged. Authorship and reuse rights of the supplied reference have not been independently established; retain attribution and check permission before public publication.

## Important assignment limitation
This is a **custom single-store, single-echelon Gymnasium simulator**, not `or-gym/InvManagement-v1`. That named environment is multi-echelon with a Box action space; this revision uses a discrete five-unit order grid. These are different models, and this notebook does not claim equivalence or instructor approval. If the assignment strictly requires `InvManagement-v1`, migration or an approved exception is still necessary.

## Run in Colab
1. Upload `Retail_Inventory_DQN_Reproducible_Study.ipynb` to Google Colab.
2. Use a free CPU runtime. Run all cells from a fresh runtime. The notebook creates its own Python source, tests and requirements; no other uploaded files are needed.
3. The pinned dependency cell must complete before imports. It installs CPU PyTorch from its official index and other packages from PyPI. No Drive mount, account secrets or remote publication is used.
4. Read the environment test output and measured 10,000-step benchmark. Default experiment: 150,000 steps each for baseline and penalty-tuned DQN, each with seeds 11, 22, 33 (900,000 training steps total, plus a discarded benchmark).
5. The final cell writes `retail_inventory_results.zip` with models, raw results, plots and this documentation. Download the executed notebook separately from Colab to retain its cell outputs.

## Measured results

Values below are from the completed Colab notebook. Profit is in the simulator's monetary units per 30-day month, evaluated under the original business costs.

| Condition | Policy | Mean monthly profit | Mean lost units |
| --- | --- | ---: | ---: |
| Normal | Tuned (s,S) heuristic | 22,615.980 | 2.920 |
| Normal | Baseline DQN | 22,343.667 | 5.020 |
| Normal | 10× penalty DQN | 22,046.367 | 2.647 |
| Stress | Tuned (s,S) heuristic | 18,444.400 | 103.340 |
| Stress | Baseline DQN | 18,333.207 | 104.163 |
| Stress | 10× penalty DQN | 18,237.027 | 97.457 |

The heuristic had the highest mean profit in both conditions. Stronger shortage penalties reduced lost units compared with baseline DQN, at the cost of higher holding costs and lower profit. These are simulated results, not evidence of real-world profitability. See the notebook's per-seed results and paired intervals for uncertainty and variability.

## Modeling choices
- 30-day episodes; mean daily demand 20, IID Poisson; unannounced stress demand mean 60 on days 15–17.
- Initial stock 40, two-day lead time, no opening pipeline, capacity 250, maximum daily order 120. Orders are selected in five-unit increments. Accepted quantity may be smaller when capacity binds.
- Decisions use current on-hand, each pipeline bucket and time remaining, all scaled to [0,1]. Previous IID demand adds no true predictive signal and is omitted.
- Order first, receive due stock, meet demand, charge holding. A day-t order arrives before demand on day t+2.
- Capacity is conservatively reserved for on-hand plus **all in-transit inventory**. Orders above free reserved capacity are rejected explicitly. No arrivals vanish.
- Original business economics: sale price 100, acquisition cost 60, end-of-day holding cost 2/unit/day, lost-sales penalty 50/unit. Opening inventory is purchased on day 1; terminal on-hand and pipeline inventory receive salvage credit at unit cost. Every policy gets identical treatment. This simplifying full-recovery salvage assumption is explicit; it is not a real-world cash-flow guarantee.
- DQN uses gamma=1.0 for this finite, time-observed horizon so its objective aligns with undiscounted monthly profit. DQN reward is scaled by 1,000. The improved candidate multiplies the lost-sales penalty by 10 during training only. All reported final profit uses the **original** penalty of 50. Shaped rewards must not be compared as if they were the same economic objective.

## Fairness and evaluation
- Same architecture, action/observation space, optimizer, exploration schedule, training budgets and seed list for both RL variants.
- `(s,S)` is tuned on 50 validation episodes. RL checkpoints are selected by original-cost mean validation profit every 25,000 steps.
- Final test: 100 independent 30-day demand scenarios per condition. Every policy and seed sees the identical scenario bank within a condition. The normal and stress banks are distinct; there is no claim of a paired before/after causal shock estimate.
- Test data is not used for tuning, selecting runs or changing hyperparameters. The fixed representative figure uses test episode 0 and training seed 11, rather than the most attractive trajectory.
- Aggregate RL scores average seeds and episodes. Seeds share test demand, so 300 RL outcomes are not 300 independent demand scenarios. Per-seed metrics expose training variability. Paired bootstrap intervals average seeds within each episode and quantify demand-scenario uncertainty **conditional on these trained policies**, not full training-seed uncertainty.
- A larger penalty is a tested improvement candidate, not a promised win. Report unfavorable or mixed outcomes honestly. Stress recovery and stockout avoidance must be established from measured metrics, not asserted.

## Artifact availability

Included in this repository: the executed notebook with visible tables and plots, runnable experiment source, environment regression tests, pinned requirements, and this README.

**Saved model files, raw CSV results, standalone figures, runtime manifest, and the results ZIP are not included yet.** The completed Colab notebook records creation of a 2.2 MB results archive, but that archive has not been recovered from the original runtime. The recorded outputs are preserved in the notebook; they are not a substitute for the missing raw artifacts. Additional artifacts will be added only if recovered and verified. Running the self-contained notebook in a fresh Colab CPU runtime regenerates a new artifact package; new runs may differ.

## Outputs generated by a complete run
- `artifacts/models/*_best.zip` and `*_final.zip`: actual trained SB3 models
- `artifacts/metrics/`: training episodes, checkpoint validation, heuristic grid, test episode metrics, demand banks, seed summaries, paired profit differences and fixed representative trajectories
- `artifacts/figures/`: training, validation, normal/stress comparison, stress trajectory
- `artifacts/run_manifest.json`: actual runtime versions, device, seeds, timesteps and measured training durations
- `artifacts/environment_tests.txt`: actual pytest result

## Current verification status
The executed Colab notebook records successful environment tests (84 passed), six 150,000-step training runs, and evaluation on 100 matched 30-day scenarios per condition. The heuristic achieved the highest mean profit. The 10× stockout-penalty DQN reduced lost units relative to baseline DQN but also reduced original-cost profit. No superior-profit claim is made. Exact visible evaluation metrics are preserved in the executed notebook. The separate result artifacts and runtime manifest are pending recovery.

## Remaining limitations
Custom environment differs from required or-gym; finite horizon and full-cost salvage simplify terminal economics; demand parameters and costs are stylized; no backorders, supplier uncertainty, nonstationary training, multi-product constraints or real data. Three training seeds are modest. A good simulator score is not evidence of safe retail deployment. This revision provides the experiment and a concise interpretation guide, not the later final written project report.
