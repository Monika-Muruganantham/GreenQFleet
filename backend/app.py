"""
GreenQ-Fleet API -- FastAPI backend for SIH26138.

Wires together: data layer -> fuel prediction model -> quantum-inspired
optimizer -> classical baselines -> scenario analysis, and serves the
frontend as static files so the whole demo runs from a single process.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.data.generate_data import load_vessels, load_fuels, ensure_dataset
from backend.prediction.fuel_model import FuelPredictor
from backend.optimization.objectives import FleetEvaluator
from backend.optimization.qiga import QIGAOptimizer
from backend.optimization.classical import classical_genetic_algorithm, random_search, greedy_heuristic, run_with_timing
from backend.optimization.scenario import run_scenario_analysis

DATA_PATH = ROOT / "data" / "operational_history.csv"

# ---- Load data & train the prediction model once at startup ----
vessels_df = load_vessels()
fuels_df = load_fuels()
operational_df = ensure_dataset(DATA_PATH)
predictor = FuelPredictor()
train_metrics = predictor.fit(operational_df)

app = FastAPI(title="GreenQ-Fleet API", version="1.0.0",
              description="Quantum-Inspired Fuel Consumption Prediction and Green Fleet Optimization (SIH26138)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                    allow_methods=["*"], allow_headers=["*"])


class Scenario(BaseModel):
    cargo_demand_t: float = Field(20000, gt=0)
    distance_nm: float = Field(1200, gt=0)
    deadline_hours: float = Field(96, gt=0)
    max_emissions_t: float = Field(200, gt=0)
    fuel_weight: float = Field(35, ge=0)
    cost_weight: float = Field(30, ge=0)
    emission_weight: float = Field(35, ge=0)
    allow_alt_fuels: bool = True
    shore_power_required: bool = False
    budget: float | None = Field(None, gt=0)
    population_size: int = Field(24, ge=10, le=200)
    generations: int = Field(40, ge=10, le=300)

    def weights(self):
        total = self.fuel_weight + self.cost_weight + self.emission_weight
        if total <= 0:
            return (1 / 3, 1 / 3, 1 / 3)
        return (self.fuel_weight / total, self.cost_weight / total, self.emission_weight / total)


class PredictionRequest(BaseModel):
    vessel_type: str
    speed_kn: float = Field(..., gt=0)
    cargo_load_t: float = Field(..., ge=0)
    distance_nm: float = Field(..., gt=0)
    fuel_type: str
    weather: str = "Moderate"


def build_evaluator(s: Scenario) -> FleetEvaluator:
    w_fuel, w_cost, w_emission = s.weights()
    return FleetEvaluator(
        vessels_df, fuels_df, predictor,
        cargo_demand_t=s.cargo_demand_t, distance_nm=s.distance_nm,
        deadline_hours=s.deadline_hours, max_emissions_t=s.max_emissions_t,
        w_fuel=w_fuel, w_cost=w_cost, w_emission=w_emission,
        allow_alt_fuels=s.allow_alt_fuels, shore_power_required=s.shore_power_required,
        budget=s.budget,
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "operational_records": len(operational_df),
            "vessels": len(vessels_df), "fuel_types": len(fuels_df),
            "model_metrics": train_metrics}


@app.get("/api/vessels")
def get_vessels():
    return vessels_df.to_dict(orient="records")


@app.get("/api/fuels")
def get_fuels():
    return fuels_df.to_dict(orient="records")


@app.get("/api/model-metrics")
def model_metrics():
    fi = predictor.feature_importance().to_dict(orient="records")
    return {"metrics": train_metrics, "feature_importance": fi}


@app.post("/api/predict")
def predict(req: PredictionRequest):
    if req.vessel_type not in vessels_df.vessel_type.values:
        raise HTTPException(400, f"Unknown vessel_type '{req.vessel_type}'")
    if req.fuel_type not in fuels_df.fuel_type.values:
        raise HTTPException(400, f"Unknown fuel_type '{req.fuel_type}'")
    fuel_l = predictor.predict_one(req.vessel_type, req.speed_kn, req.cargo_load_t,
                                    req.distance_nm, req.fuel_type, req.weather)
    fuel_row = fuels_df[fuels_df.fuel_type == req.fuel_type].iloc[0]
    cost = fuel_l * fuel_row.price_per_litre
    emissions_t = fuel_l * fuel_row.lifecycle_kgco2_per_litre / 1000.0
    voyage_hours = req.distance_nm / req.speed_kn
    return {"fuel_l": round(fuel_l, 1), "cost": round(cost, 2),
            "emissions_t": round(emissions_t, 3), "voyage_hours": round(voyage_hours, 1)}


@app.post("/api/optimize")
def optimize(s: Scenario):
    evaluator = build_evaluator(s)
    result = QIGAOptimizer(evaluator, population_size=s.population_size, generations=s.generations, seed=42).run()
    return result


@app.post("/api/benchmark")
def benchmark(s: Scenario):
    evaluator = build_evaluator(s)
    qiga = run_with_timing(lambda: QIGAOptimizer(evaluator, s.population_size, s.generations, 42).run())
    ga = run_with_timing(classical_genetic_algorithm, evaluator, s.population_size, s.generations, 42)
    rs = run_with_timing(random_search, evaluator, trials=s.population_size * s.generations, seed=42)
    greedy = run_with_timing(greedy_heuristic, evaluator)

    def summarize(r):
        return {"method": r["method"], "fitness": round(r["fitness"], 5), "fuel_l": round(r["fuel_l"], 1),
                "cost": round(r["cost"], 2), "emissions_t": round(r["emissions_t"], 3),
                "cargo_served_t": round(r["cargo_served_t"], 1), "feasible": r["feasible"],
                "runtime_seconds": r["runtime_seconds"], "convergence_history": r["convergence_history"]}

    return {"results": [summarize(qiga), summarize(ga), summarize(rs), summarize(greedy)]}


@app.post("/api/scenario-analysis")
def scenario_analysis(s: Scenario):
    results = run_scenario_analysis(
        vessels_df, fuels_df, predictor, s.cargo_demand_t, s.distance_nm,
        s.deadline_hours, s.max_emissions_t, s.weights(), s.budget,
        population_size=min(s.population_size, 28), generations=min(s.generations, 45),
    )
    return {"scenarios": results}


# ---- Serve the frontend (single-process demo) ----
FRONTEND_DIR = ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/style.css")
    def style():
        return FileResponse(FRONTEND_DIR / "style.css")

    @app.get("/app.js")
    def script():
        return FileResponse(FRONTEND_DIR / "app.js")
