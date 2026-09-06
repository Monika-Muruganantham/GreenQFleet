"""
Scenario analysis for alternative fuels and shore power (part of the
"Expected Solution": scenario analysis for alternative fuels).
Runs the same QIGA optimizer under a handful of policy-relevant scenarios
so a user can see how the recommended plan shifts.
"""
from __future__ import annotations
from backend.optimization.objectives import FleetEvaluator
from backend.optimization.qiga import QIGAOptimizer

SCENARIOS = [
    {"name": "Conventional fuel only",           "allow_alt_fuels": False, "shore_power_required": False, "price_multiplier": 1.0, "demand_multiplier": 1.0},
    {"name": "Alternative fuels enabled",         "allow_alt_fuels": True,  "shore_power_required": False, "price_multiplier": 1.0, "demand_multiplier": 1.0},
    {"name": "Alt fuels + mandatory shore power", "allow_alt_fuels": True,  "shore_power_required": True,  "price_multiplier": 1.0, "demand_multiplier": 1.0},
    {"name": "Bunker price shock (+25%)",         "allow_alt_fuels": True,  "shore_power_required": False, "price_multiplier": 1.25, "demand_multiplier": 1.0},
    {"name": "Peak season demand (+30%)",         "allow_alt_fuels": True,  "shore_power_required": False, "price_multiplier": 1.0, "demand_multiplier": 1.3},
]


def run_scenario_analysis(vessels, fuels, predictor, cargo_demand_t, distance_nm,
                           deadline_hours, max_emissions_t, weights, budget=None,
                           population_size: int = 24, generations: int = 40):
    w_fuel, w_cost, w_emission = weights
    results = []
    for sc in SCENARIOS:
        evaluator = FleetEvaluator(
            vessels, fuels, predictor,
            cargo_demand_t=cargo_demand_t * sc["demand_multiplier"],
            distance_nm=distance_nm, deadline_hours=deadline_hours,
            max_emissions_t=max_emissions_t, w_fuel=w_fuel, w_cost=w_cost, w_emission=w_emission,
            allow_alt_fuels=sc["allow_alt_fuels"], shore_power_required=sc["shore_power_required"],
            budget=budget, price_multiplier=sc["price_multiplier"],
        )
        outcome = QIGAOptimizer(evaluator, population_size=population_size, generations=generations, seed=7).run()
        results.append({
            "scenario": sc["name"],
            "fuel_l": outcome["fuel_l"],
            "cost": outcome["cost"],
            "emissions_t": outcome["emissions_t"],
            "cargo_served_t": outcome["cargo_served_t"],
            "vessels_used": outcome["vessels_used"],
            "feasible": outcome["feasible"],
        })
    return results
