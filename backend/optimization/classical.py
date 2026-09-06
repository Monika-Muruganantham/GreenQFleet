"""
Classical baseline optimizers, used purely for benchmarking against the
quantum-inspired optimizer (Objective 5 of SIH26138: benchmark accuracy,
convergence speed, solution quality, and scalability against conventional
methods). Same evaluator, same search space -- only the search strategy
differs, so comparisons are fair.
"""
from __future__ import annotations
import time
import numpy as np


def classical_genetic_algorithm(evaluator, population_size: int = 40, generations: int = 80, seed: int = 42):
    """Standard real-valued GA: tournament-free elitism + uniform crossover + mutation."""
    rng = np.random.default_rng(seed)
    n = evaluator.n_genes
    population = [rng.random(n) for _ in range(population_size)]
    best = (1e99, None)
    history = []
    for _ in range(generations):
        scored = sorted(((evaluator.evaluate(x), x) for x in population), key=lambda z: z[0])
        if scored[0][0] < best[0]:
            best = (scored[0][0], scored[0][1].copy())
        history.append(float(best[0]))
        elites = [x for _, x in scored[:max(4, population_size // 5)]]
        children = []
        while len(children) < population_size:
            a, b = elites[rng.integers(len(elites))], elites[rng.integers(len(elites))]
            cut = rng.integers(1, n - 1)
            child = np.r_[a[:cut], b[cut:]].copy()
            mutate_mask = rng.random(n) < 0.05
            child[mutate_mask] = rng.random(int(mutate_mask.sum()))
            children.append(child)
        population = children
    result = evaluator.evaluate(best[1], return_details=True)
    result["convergence_history"] = history
    result["method"] = "Classical Genetic Algorithm"
    return result


def random_search(evaluator, trials: int = 3200, seed: int = 42):
    """Pure random search -- the weakest reasonable baseline, establishes a floor."""
    rng = np.random.default_rng(seed)
    n = evaluator.n_genes
    best = (1e99, None)
    history = []
    for _ in range(trials):
        x = rng.random(n)
        f = evaluator.evaluate(x)
        if f < best[0]:
            best = (f, x)
        history.append(float(best[0]))
    result = evaluator.evaluate(best[1], return_details=True)
    result["convergence_history"] = history[::max(1, trials // 80)]  # thin for chart display
    result["method"] = "Random Search"
    return result


def greedy_heuristic(evaluator):
    """
    Deterministic rule-of-thumb a human planner might use: pick vessels in
    capacity order, run at mid-range speed, always use conventional fuel,
    no shore power. This represents "how it's done today" without any
    optimization -- the honest baseline for headline fuel/emission savings.
    """
    vessels = evaluator.vessels.sort_values("capacity_tonnes", ascending=False)
    remaining = evaluator.cargo_demand_t
    n = len(evaluator.vessels)
    genome = np.zeros(evaluator.n_genes)
    conventional_idx = 0
    for idx in evaluator.fuels.index:
        if "Conventional" in evaluator.fuels.loc[idx, "fuel_type"]:
            conventional_idx = idx
            break
    fuel_frac = (conventional_idx + 0.5) / len(evaluator.fuels)

    for i, v in evaluator.vessels.iterrows():
        if remaining <= 0:
            break
        genome[i] = 1.0  # use this vessel
        genome[n + i] = 0.5  # mid-range speed
        genome[2 * n + i] = ((min(v.capacity_tonnes, remaining) / v.capacity_tonnes) - 0.30) / 0.70
        genome[3 * n + i] = fuel_frac
        genome[4 * n + i] = 0.0  # no shore power
        remaining -= v.capacity_tonnes

    result = evaluator.evaluate(np.clip(genome, 0, 1), return_details=True)
    result["convergence_history"] = [result["fitness"]]
    result["method"] = "Greedy Heuristic (status quo)"
    return result


def run_with_timing(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    result["runtime_seconds"] = round(time.perf_counter() - t0, 4)
    return result
