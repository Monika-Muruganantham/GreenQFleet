"""
Quantum-Inspired Genetic Algorithm (QIGA) for green fleet deployment
(Objective 2 of SIH26138 -- the platform's core novelty).

This borrows the qubit-encoding / rotation-gate structure of Quantum-Inspired
Evolutionary Algorithms (Han & Kim, 2002) and runs it on classical hardware
-- it does NOT require or simulate a real quantum processor. Each decision
gene is represented as a probability amplitude q in [0,1] (the |1> observation
probability of a conceptual qubit, analogous to alpha^2/beta^2 amplitudes).
Each generation:
  1. Observe: sample a classical candidate solution from the probability
     vector (this is the "measurement" step).
  2. Evaluate the candidate against the multi-objective fitness.
  3. Rotate: nudge each qubit's probability toward the best-known solution,
     analogous to a quantum rotation gate steering amplitude toward the
     |1> or |0> basis state correlated with better fitness.

This gives the algorithm two properties classical GAs don't have as
naturally: (a) implicit population diversity from a single probability
distribution rather than an explicit population needing separate mutation
tuning, and (b) faster convergence on rugged, high-dimensional landscapes
because the whole probability vector moves each generation rather than only
surviving individuals -- this is what gets benchmarked against classical
GA / random search in classical.py.
"""
from __future__ import annotations
import numpy as np


class QIGAOptimizer:
    def __init__(self, evaluator, population_size: int = 40, generations: int = 80, seed: int = 42):
        self.evaluator = evaluator
        self.population_size = population_size
        self.generations = generations
        self.rng = np.random.default_rng(seed)
        self.n = evaluator.n_genes

    def _observe(self, q: np.ndarray) -> np.ndarray:
        """Collapse each qubit's probability amplitude into a classical candidate value."""
        binary = (self.rng.random(self.n) < q).astype(float)
        # blend with q itself so genes carry continuous information (speed/load fractions),
        # not just a 0/1 collapse -- needed since our decode step uses continuous genes.
        return binary * 0.75 + q * 0.25

    def run(self):
        q = np.full(self.n, 0.55)
        best = None
        history = []
        for g in range(self.generations):
            population = [self._observe(q) for _ in range(self.population_size)]
            for x in population:  # small mutation to preserve exploration
                mask = self.rng.random(self.n) < 0.04
                x[mask] = self.rng.random(int(mask.sum()))

            scored = sorted(((self.evaluator.evaluate(x), x) for x in population), key=lambda z: z[0])
            if best is None or scored[0][0] < best[0]:
                best = (scored[0][0], scored[0][1].copy())

            # Quantum rotation-inspired update: move probability vector toward elite
            rotation_rate = max(0.015, 0.12 * (1 - g / self.generations))
            q += rotation_rate * (best[1] - q)
            q = np.clip(q, 0.03, 0.97)
            history.append(float(best[0]))

        result = self.evaluator.evaluate(best[1], return_details=True)
        result["convergence_history"] = history
        result["method"] = "Quantum-Inspired GA (QIGA)"
        return result
