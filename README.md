# GreenQ-Fleet — Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization

**SIH26138** · Prototype software platform

A working demo that predicts vessel fuel consumption from operating conditions and optimizes
green fleet deployment — vessel mix, cargo split, cruising speed, alternative fuel choice
(LNG / methanol / hydrogen / ammonia), and shore power — using a **quantum-inspired
metaheuristic** (Q-bit rotation evolutionary search) running on ordinary CPUs, benchmarked
live against classical optimization methods.

---

## Quick start

```bash
pip install -r requirements.txt
python run.py
```

This starts a single local server at **http://127.0.0.1:8000** and opens it in your browser
automatically. No separate frontend build step, no external services, no API keys, no internet
connection required at runtime (Chart.js loads from a CDN in the browser — everything else runs
locally).

First launch trains the prediction model on ~4,000 synthetic operational records (a few
seconds); this is cached in memory for the life of the process.

**Requirements:** Python 3.10+.

---

## What's inside

```
GreenQFleet/
├── run.py                          # one-command launcher (starts server, opens browser)
├── requirements.txt
├── backend/
│   ├── app.py                      # FastAPI app — all /api/* endpoints + serves the frontend
│   ├── data/
│   │   └── generate_data.py        # vessel & fuel master data + synthetic operational dataset
│   ├── prediction/
│   │   └── fuel_model.py           # Gradient Boosting fuel consumption predictor
│   └── optimization/
│       ├── objectives.py           # multi-objective fitness + constraint handling (decode/evaluate)
│       ├── qiga.py                 # Quantum-Inspired Genetic Algorithm (the core novelty)
│       ├── classical.py            # classical GA, random search, greedy baseline — for benchmarking
│       └── scenario.py             # alternative-fuel / shore-power scenario analysis
├── frontend/
│   ├── index.html                  # single-page dashboard (5 tabs)
│   ├── style.css                   # dark maritime/green theme
│   └── app.js                      # tab logic, API calls, Chart.js visualizations
└── data/
    └── operational_history.csv     # generated on first run (synthetic training data)
```

## Architecture

```
Operational Data  →  Fuel Prediction (ML)  →  Quantum-Inspired Optimizer  →  Fleet Deployment Plan
   (synthetic)         Gradient Boosting        Q-bit rotation search           + benchmarking
                        R² ≈ 0.98 holdout        vs. classical GA /             + scenario analysis
                                                  random search / greedy
```

**Why the optimizer is "quantum-inspired" and not just another GA:** every deployment decision
(use vessel, speed, cargo split, fuel choice, shore power) is encoded as a qubit-style probability
amplitude in `[0,1]`. Each generation *samples* ("observes") a candidate fleet plan from that
probability field, evaluates it against the multi-objective fitness, then *rotates* the entire
probability field toward the best-known solution — steering the whole search distribution at
once each generation, rather than only propagating surviving individuals as a classical GA does.
This borrows the structure of Quantum-Inspired Evolutionary Algorithms (Han & Kim, 2002) and runs
entirely on classical hardware — it does not require or simulate real quantum hardware, and the
app is explicit about that in the UI so the claim stays honest and defensible in front of judges.

This claim is not just asserted — the **Benchmarking** tab runs the identical objective and
constraints through a classical GA, pure random search, and a greedy "status quo" heuristic, live,
so you can see the actual convergence speed and solution quality difference on your machine.

---

## Delivery Table (Expected Deliverables)

| # | Deliverable (from problem statement) | Where it lives in this prototype | Status |
|---|---|---|---|
| 1 | Mathematical modelling of fuel consumption vs. speed, load, distance, sea state, fuel type | `backend/data/generate_data.py` (synthetic physics-motivated generator) | ✅ Included |
| 2 | Data-driven fuel consumption prediction module, accurate across vessel types & conditions | `backend/prediction/fuel_model.py` — **Fuel Prediction** tab (R² ≈ 0.98, MAPE ≈ 14% on holdout) | ✅ Included |
| 3 | Quantum-inspired metaheuristic optimization framework for vessel mix, capacity, cruising speed | `backend/optimization/qiga.py` — **Fleet Optimization** tab | ✅ Included |
| 4 | Multi-objective minimization: fuel, operational cost, lifecycle GHG emissions | `backend/optimization/objectives.py` (weighted, user-adjustable) | ✅ Included |
| 5 | Constraint handling: cargo demand, schedule reliability, emission regulation compliance | Penalty-based constraint handling in `objectives.py`; feasibility flag + plain-language reasons in the UI | ✅ Included |
| 6 | Alternative fuel integration: LNG, methanol, hydrogen, ammonia | `backend/data/generate_data.py` fuel master table; selectable everywhere | ✅ Included |
| 7 | Shore power integration | Cold-ironing fuel discount + shore-power-compatible fuel flag in `objectives.py` | ✅ Included |
| 8 | Scenario analysis for alternative fuels & operating conditions | `backend/optimization/scenario.py` — **Scenario Analysis** tab (5 policy scenarios) | ✅ Included |
| 9 | Benchmarking vs. conventional methods — accuracy, convergence speed, solution quality, scalability | `backend/optimization/classical.py` — **Benchmarking** tab (same objective, 4 methods, live timing + convergence charts) | ✅ Included |
| 10 | Comprehensive software platform / case-study-ready UI | This app — FastAPI backend + custom dashboard, 5 tabs, interactive controls | ✅ Included |
| 11 | Real-world AIS / noon-report / bunker-price data integration | Synthetic generator is a drop-in placeholder with the same schema | 🔜 Future work |
| 12 | Execution on real quantum hardware / QPU-backed QAOA | Currently quantum-inspired on classical hardware only, by design (see note above) | 🔜 Future work |
| 13 | IMO/EU MRV-calibrated lifecycle emission factors, regulatory compliance certification | Emission factors are illustrative, not certified | 🔜 Future work |

---

## Honest scope notes (say this out loud to judges before they ask)

- **Data is synthetic.** The operational dataset is generated from a physically-motivated but
  illustrative formula (`backend/data/generate_data.py`), not real AIS/noon-report data. Fuel
  values are in consistent internal units for comparison purposes, not calibrated bunker-fuel
  scales.
- **"Quantum-inspired" means classical hardware.** No quantum processor is used or simulated.
  The novelty is the qubit-rotation search *strategy*, which is benchmarked, not assumed, against
  classical alternatives in the app itself.
- **Prediction is intentionally classical ML** (Gradient Boosting) — that's the right engineering
  choice for a supervised regression problem; the quantum-inspired contribution is scoped to the
  combinatorial optimization stage where it actually helps.

## Extending toward production

1. Replace `generate_operational_data()` with a loader for real noon-report / AIS / bunker-price
   feeds — the downstream prediction and optimization code needs no changes since it only depends
   on the same column schema.
2. Swap emission factors for IMO/EU MRV-published lifecycle factors per fuel pathway.
3. Add a real multi-voyage / multi-port routing layer (this prototype optimizes a single
   demand/distance scenario at a time — extending to a rolling schedule is the natural next step).
4. For teams with quantum hardware access, `qiga.py`'s `_observe`/rotation step can be swapped for
   an actual QAOA circuit (e.g. via Qiskit) evaluating the same objective as a real-hardware
   variant to compare against the classical-hardware quantum-inspired baseline.
