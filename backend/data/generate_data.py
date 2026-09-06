"""
Maritime fleet & fuel master data, and synthetic operational dataset generation.

In production these would be pulled from: vessel particulars databases (IMO
records), noon reports / AIS-derived speed logs, bunker fuel pricing feeds,
and IMO/EU MRV lifecycle emission factors. For this prototype we generate a
physically-motivated synthetic dataset so prediction + optimization can run
end-to-end without needing a live data contract.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42

# vessel_id, type, capacity (tonnes), engine power (kW), min/max cruising speed (knots)
VESSEL_TYPES = [
    ("V001", "Feeder",    3000,  1800, 10, 18),
    ("V002", "Feeder",    3500,  2000, 10, 18),
    ("V003", "Panamax",   8000,  7000, 11, 21),
    ("V004", "Panamax",   9000,  7600, 11, 21),
    ("V005", "Handymax",  6000,  4800, 10, 19),
    ("V006", "Handymax",  6500,  5000, 10, 19),
    ("V007", "Container", 12000, 10000, 12, 23),
    ("V008", "Container", 14000, 11500, 12, 23),
    ("V009", "Ro-Ro",     7000,  5200, 11, 20),
    ("V010", "Ro-Ro",     8000,  6000, 11, 20),
    ("V011", "Tanker",    10000, 8500, 10, 19),
    ("V012", "Tanker",    12000, 10000, 10, 19),
]

# price (currency units per litre-equivalent), lifecycle kgCO2e per litre-equivalent,
# relative energy efficiency factor, shore-power compatible flag
FUEL_TYPES = [
    ("Conventional (HFO/MGO)", 92.0,  3.20, 1.00, True),
    ("LNG",                    78.0,  2.35, 0.96, True),
    ("Methanol",               84.0,  1.95, 1.04, True),
    ("Hydrogen",               132.0, 0.75, 0.90, True),
    ("Ammonia",                102.0, 1.10, 1.06, False),
]


def load_vessels() -> pd.DataFrame:
    rows = [
        {"vessel_id": vid, "vessel_type": typ, "capacity_tonnes": cap,
         "engine_power_kw": power, "min_speed_kn": mn, "max_speed_kn": mx}
        for vid, typ, cap, power, mn, mx in VESSEL_TYPES
    ]
    return pd.DataFrame(rows)


def load_fuels() -> pd.DataFrame:
    rows = [
        {"fuel_type": name, "price_per_litre": price, "lifecycle_kgco2_per_litre": co2,
         "efficiency_factor": eff, "shore_power_compatible": shore}
        for name, price, co2, eff, shore in FUEL_TYPES
    ]
    return pd.DataFrame(rows)


def generate_operational_data(vessels: pd.DataFrame, fuels: pd.DataFrame, n: int = 4000, seed: int = RNG_SEED) -> pd.DataFrame:
    """
    Synthetic but physically-motivated: fuel burn rises steeply (~cube law)
    with speed, scales with engine power and distance, is nudged by cargo
    load and sea state, and varies with fuel type's relative efficiency.
    Deliberately noisy and nonlinear so a real model has structure to learn
    rather than a trivial rule.
    """
    rng = np.random.default_rng(seed)
    weather_states = ["Calm", "Moderate", "Rough"]
    weather_factor = {"Calm": 0.94, "Moderate": 1.00, "Rough": 1.16}

    records = []
    for _ in range(n):
        v = vessels.iloc[rng.integers(len(vessels))]
        f = fuels.iloc[rng.integers(len(fuels))]
        speed = rng.uniform(v.min_speed_kn, v.max_speed_kn)
        load_frac = rng.uniform(0.2, 1.0)
        load = load_frac * v.capacity_tonnes
        distance = rng.uniform(150, 2200)  # nautical miles
        weather = rng.choice(weather_states, p=[0.45, 0.4, 0.15])

        base = (v.engine_power_kw / 1000.0) * distance / 68.0
        speed_factor = (speed / 15.0) ** 2.75          # admiralty-style cube-ish law
        load_factor = 0.80 + 0.35 * load_frac
        fuel = (base * speed_factor * load_factor
                * weather_factor[weather] * f.efficiency_factor
                * rng.normal(1.0, 0.045))
        fuel = max(fuel, 50.0)

        records.append({
            "vessel_type": v.vessel_type,
            "speed_kn": round(speed, 2),
            "cargo_load_t": round(load, 1),
            "distance_nm": round(distance, 1),
            "fuel_type": f.fuel_type,
            "weather": weather,
            "fuel_consumption_l": round(fuel, 1),
        })
    return pd.DataFrame(records)


def ensure_dataset(path: Path, n: int = 4000) -> pd.DataFrame:
    """Generate the operational dataset if it doesn't already exist on disk, else load it."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vessels = load_vessels()
    fuels = load_fuels()
    df = generate_operational_data(vessels, fuels, n=n)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "data" / "operational_history.csv"
    df = ensure_dataset(out)
    print(f"Dataset ready: {len(df)} rows -> {out}")
