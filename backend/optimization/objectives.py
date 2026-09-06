"""
Multi-objective evaluation and constraint handling for green fleet deployment
decisions (Objectives 2-4 of SIH26138).

A candidate solution ("genome") is a vector of continuous values in [0,1]
per vessel, decoded into: use/don't-use, cruising speed within the vessel's
operating range, cargo load fraction, choice of fuel type, and shore-power
usage. This decode step is what both the quantum-inspired optimizer (qiga.py)
and the classical baselines (classical.py) evaluate identically, so the
comparison in the Benchmarking tab is apples-to-apples -- only the *search
strategy* differs between methods, not the objective being optimized.

Constraints (cargo demand satisfaction, schedule reliability / deadline,
emissions cap, optional budget cap, optional shore-power mandate) are
handled via penalty terms added to the scalarized multi-objective fitness,
the standard approach for metaheuristic (GA / quantum-inspired / annealing)
optimizers that don't have a native constraint-handling mechanism.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

GENES_PER_VESSEL = 5  # [use, speed_frac, load_frac, fuel_choice, shore_power]


class FleetEvaluator:
    def __init__(self, vessels: pd.DataFrame, fuels: pd.DataFrame, predictor,
                 cargo_demand_t: float, distance_nm: float, deadline_hours: float,
                 max_emissions_t: float,
                 w_fuel: float = 0.35, w_cost: float = 0.30, w_emission: float = 0.35,
                 allow_alt_fuels: bool = True, shore_power_required: bool = False,
                 budget: float | None = None, price_multiplier: float = 1.0):
        self.vessels = vessels.reset_index(drop=True)
        self.fuels = fuels.reset_index(drop=True) if allow_alt_fuels else fuels[fuels.fuel_type.str.contains("Conventional")].reset_index(drop=True)
        self.predictor = predictor
        self.cargo_demand_t = cargo_demand_t
        self.distance_nm = distance_nm
        self.deadline_hours = deadline_hours
        self.max_emissions_t = max_emissions_t
        total_w = max(w_fuel + w_cost + w_emission, 1e-9)
        self.w_fuel, self.w_cost, self.w_emission = w_fuel / total_w, w_cost / total_w, w_emission / total_w
        self.shore_power_required = shore_power_required
        self.budget = budget
        self.price_multiplier = price_multiplier
        self.n_genes = GENES_PER_VESSEL * len(self.vessels)

    def decode(self, genome: np.ndarray) -> list[dict]:
        n = len(self.vessels)
        decisions = []
        for i, v in self.vessels.iterrows():
            use = genome[i] > 0.30
            if not use:
                continue
            speed = v.min_speed_kn + float(genome[n + i]) * (v.max_speed_kn - v.min_speed_kn)
            load_frac = 0.30 + 0.70 * float(genome[2 * n + i])
            cargo = min(v.capacity_tonnes * load_frac, v.capacity_tonnes)
            fuel_idx = min(len(self.fuels) - 1, int(float(genome[3 * n + i]) * len(self.fuels)))
            fuel_row = self.fuels.iloc[fuel_idx]
            shore = (float(genome[4 * n + i]) > 0.5 and bool(fuel_row.shore_power_compatible)) or self.shore_power_required
            decisions.append({"vessel": v, "speed": speed, "cargo": cargo, "fuel_row": fuel_row, "shore": shore})
        return decisions

    def evaluate(self, genome: np.ndarray, return_details: bool = False):
        decisions = self.decode(genome)
        if not decisions:
            return {"fitness": 1e9} if return_details else 1e9

        # Batch all per-vessel predictions into a single model call instead of
        # one call per vessel -- this is what keeps optimizer generations fast
        # enough for interactive use as the fleet/job count scales up.
        batch_df = pd.DataFrame([{
            "vessel_type": d["vessel"].vessel_type, "fuel_type": d["fuel_row"].fuel_type,
            "weather": "Moderate", "speed_kn": d["speed"], "cargo_load_t": d["cargo"],
            "distance_nm": self.distance_nm,
        } for d in decisions])
        fuel_preds = self.predictor.predict_batch(batch_df)

        total_fuel = total_cost = total_emissions = served = 0.0
        max_voyage_hours = 0.0
        rows = []
        for d, fuel_l in zip(decisions, fuel_preds):
            v, speed, cargo, fuel_row, shore = d["vessel"], d["speed"], d["cargo"], d["fuel_row"], d["shore"]
            if shore:
                fuel_l *= 0.965  # cold-ironing at berth reduces auxiliary engine burn
            cost = fuel_l * fuel_row.price_per_litre * self.price_multiplier
            emissions_t = fuel_l * fuel_row.lifecycle_kgco2_per_litre / 1000.0
            voyage_hours = self.distance_nm / max(speed, 1e-6)

            total_fuel += fuel_l
            total_cost += cost
            total_emissions += emissions_t
            served += cargo
            max_voyage_hours = max(max_voyage_hours, voyage_hours)

            rows.append({
                "Vessel": v.vessel_id, "Type": v.vessel_type, "Cargo (t)": round(cargo, 1),
                "Speed (kn)": round(speed, 2), "Fuel": fuel_row.fuel_type,
                "Shore Power": "Yes" if shore else "No",
                "Fuel (L)": round(fuel_l, 1), "CO2 (t)": round(emissions_t, 3),
                "Cost": round(cost, 2), "Voyage (h)": round(voyage_hours, 1),
            })

        penalty = 0.0
        if served < self.cargo_demand_t:
            penalty += (self.cargo_demand_t - served) * 12.0
        if max_voyage_hours > self.deadline_hours:
            penalty += (max_voyage_hours - self.deadline_hours) * 800.0
        if total_emissions > self.max_emissions_t:
            penalty += (total_emissions - self.max_emissions_t) * 60.0
        if self.budget is not None and total_cost > self.budget:
            penalty += (total_cost - self.budget) * 0.05

        fitness = (self.w_fuel * (total_fuel / 12000.0)
                   + self.w_cost * (total_cost / 1_200_000.0)
                   + self.w_emission * (total_emissions / 120.0)
                   + penalty / 1e6)

        if not return_details:
            return fitness

        reasons = []
        reasons.append("Cargo demand satisfied." if served >= self.cargo_demand_t
                        else f"Cargo shortfall of {self.cargo_demand_t - served:,.0f} t vs demand.")
        reasons.append("Schedule reliability met within deadline." if max_voyage_hours <= self.deadline_hours
                        else "Longest voyage exceeds the deadline; consider higher speeds or more vessels.")
        reasons.append("Lifecycle emissions within regulatory cap." if total_emissions <= self.max_emissions_t
                        else "Emissions cap exceeded; increase the cap or shift weight toward emissions.")
        if any(r["Shore Power"] == "Yes" for r in rows):
            reasons.append("Shore power (cold-ironing) used at berth where it improves the objective.")
        alt_used = [r["Fuel"] for r in rows if "Conventional" not in r["Fuel"]]
        if alt_used:
            reasons.append(f"Alternative fuel deployed on {len(alt_used)} vessel(s): {', '.join(sorted(set(alt_used)))}.")

        return {
            "fitness": fitness, "fuel_l": total_fuel, "cost": total_cost,
            "emissions_t": total_emissions, "cargo_served_t": served,
            "max_voyage_hours": max_voyage_hours, "vessels_used": len(rows),
            "rows": rows, "reasons": reasons, "feasible": penalty == 0.0,
        }
