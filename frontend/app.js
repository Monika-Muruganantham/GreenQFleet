const API = "";  // same-origin, served by FastAPI

const charts = {};  // keep references so we can destroy/redraw on rerun

// ---------- helpers ----------
function $(id) { return document.getElementById(id); }

function toast(msg, isError = false) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  el.classList.toggle("error", isError);
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add("hidden"), 3500);
}

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

function fmt(n, digits = 0) {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function makeChart(canvasId, config) {
  if (typeof Chart === "undefined") {
    // Chart.js failed to load (e.g. assets/chart.umd.js missing or blocked).
    // Fail gracefully instead of throwing, so the rest of the dashboard
    // (KPIs, tables, reasons) still renders and updates normally.
    console.warn("Chart.js is not loaded — skipping chart:", canvasId);
    toast("Charts unavailable (Chart.js failed to load) — numbers and tables still work.", true);
    return null;
  }
  const ctx = $(canvasId).getContext("2d");
  if (charts[canvasId]) charts[canvasId].destroy();
  Chart.defaults.color = "#a9bdcc";
  Chart.defaults.borderColor = "rgba(255,255,255,0.08)";
  charts[canvasId] = new Chart(ctx, config);
  return charts[canvasId];
}

function fillTable(tableId, columns, rows) {
  const table = $(tableId);
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  thead.innerHTML = "<tr>" + columns.map(c => `<th>${c}</th>`).join("") + "</tr>";
  tbody.innerHTML = rows.map(r => "<tr>" + columns.map(c => `<td>${r[c] ?? ""}</td>`).join("") + "</tr>").join("");
}

function currentScenario(overrides = {}) {
  return Object.assign({
    cargo_demand_t: Number($("op-demand").value),
    distance_nm: Number($("op-distance").value),
    deadline_hours: Number($("op-deadline").value),
    max_emissions_t: Number($("op-emissions").value),
    fuel_weight: Number($("op-wfuel").value),
    cost_weight: Number($("op-wcost").value),
    emission_weight: Number($("op-wemit").value),
    allow_alt_fuels: $("op-altfuels").checked,
    shore_power_required: $("op-shore").checked,
    population_size: 24,
    generations: 40,
  }, overrides);
}

// ---------- tab navigation ----------
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    $("panel-" + btn.dataset.tab).classList.add("active");
  });
});

// ---------- range label sync ----------
[["pv-speed", "pv-speed-val"], ["pv-load", "pv-load-val"],
 ["op-wfuel", "op-wfuel-val"], ["op-wcost", "op-wcost-val"], ["op-wemit", "op-wemit-val"]]
  .forEach(([input, label]) => {
    $(input).addEventListener("input", () => { $(label).textContent = $(input).value; });
  });

// ---------- bootstrap: populate vessel / fuel dropdowns ----------
async function bootstrap() {
  try {
    const [vessels, fuels] = await Promise.all([api("/api/vessels"), api("/api/fuels")]);
    $("pv-vessel").innerHTML = [...new Set(vessels.map(v => v.vessel_type))]
      .map(t => `<option>${t}</option>`).join("");
    $("pv-fuel").innerHTML = fuels.map(f => `<option>${f.fuel_type}</option>`).join("");

    const metrics = await api("/api/model-metrics");
    $("model-metrics").innerHTML = `
      <div class="row"><span>R² (holdout)</span><b>${metrics.metrics.r2}</b></div>
      <div class="row"><span>Mean absolute error</span><b>${fmt(metrics.metrics.mae_liters, 1)} L</b></div>
      <div class="row"><span>MAPE</span><b>${metrics.metrics.mape_pct}%</b></div>
      <div class="row"><span>Training records</span><b>${fmt(metrics.metrics.n_train)}</b></div>
      <div class="row"><span>Holdout records</span><b>${fmt(metrics.metrics.n_test)}</b></div>
    `;
    makeChart("chart-feature-importance", {
      type: "bar",
      data: {
        labels: metrics.feature_importance.slice(0, 8).map(f => f.feature),
        datasets: [{ label: "Importance", data: metrics.feature_importance.slice(0, 8).map(f => f.importance), backgroundColor: "#2dd4bf" }],
      },
      options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { grid: { color: "rgba(255,255,255,0.06)" } }, y: { grid: { display: false } } } },
    });
  } catch (e) {
    toast("Failed to load initial data: " + e.message, true);
  }
}
bootstrap();

// ---------- Fuel Prediction ----------
$("btn-predict").addEventListener("click", async () => {
  const btn = $("btn-predict");
  btn.disabled = true;
  try {
    const payload = {
      vessel_type: $("pv-vessel").value,
      speed_kn: Number($("pv-speed").value),
      cargo_load_t: Number($("pv-load").value),
      distance_nm: Number($("pv-distance").value),
      fuel_type: $("pv-fuel").value,
      weather: $("pv-weather").value,
    };
    const r = await api("/api/predict", "POST", payload);
    $("pred-fuel").textContent = fmt(r.fuel_l, 0) + " L";
    $("pred-cost").textContent = "₹" + fmt(r.cost, 0);
    $("pred-co2").textContent = fmt(r.emissions_t, 2) + " t";
    $("pred-time").textContent = fmt(r.voyage_hours, 1) + " h";
  } catch (e) {
    toast("Prediction failed: " + e.message, true);
  } finally {
    btn.disabled = false;
  }
});

// ---------- Fleet Optimization ----------
async function runOptimize(scenario) {
  const r = await api("/api/optimize", "POST", scenario);
  $("optimize-results").classList.remove("hidden");
  $("opt-fuel").textContent = fmt(r.fuel_l, 0) + " L";
  $("opt-cost").textContent = "₹" + fmt(r.cost, 0);
  $("opt-co2").textContent = fmt(r.emissions_t, 2) + " t";
  $("opt-cargo").textContent = fmt(r.cargo_served_t, 0) + " t";

  fillTable("opt-table",
    ["Vessel", "Type", "Cargo (t)", "Speed (kn)", "Fuel", "Shore Power", "Fuel (L)", "CO2 (t)", "Cost"],
    r.rows);

  $("opt-reasons").innerHTML = r.reasons.map(x => `<li>${x}</li>`).join("");

  makeChart("chart-convergence", {
    type: "line",
    data: {
      labels: r.convergence_history.map((_, i) => i + 1),
      datasets: [{ label: "Best fitness", data: r.convergence_history, borderColor: "#2dd4bf", backgroundColor: "rgba(45,212,191,0.15)", fill: true, tension: 0.25, pointRadius: 0 }],
    },
    options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "Generation" } }, y: { title: { display: true, text: "Fitness (lower = better)" } } } },
  });
  return r;
}

$("btn-optimize").addEventListener("click", async () => {
  const btn = $("btn-optimize");
  btn.disabled = true;
  btn.textContent = "Optimizing…";
  try {
    await runOptimize(currentScenario());
    toast("Optimization complete.");
  } catch (e) {
    toast("Optimization failed: " + e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "🚀 Optimize Fleet";
  }
});

// ---------- Overview: quick baseline vs optimized ----------
$("btn-run-overview").addEventListener("click", async () => {
  const btn = $("btn-run-overview");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const scenario = currentScenario();
    const [bench] = await Promise.all([api("/api/benchmark", "POST", Object.assign({}, scenario, { population_size: 20, generations: 25 }))]);
    const qiga = bench.results.find(r => r.method.includes("Quantum"));
    const greedy = bench.results.find(r => r.method.includes("Greedy"));

    $("kpi-baseline-fuel").textContent = fmt(greedy.fuel_l, 0) + " L";
    $("kpi-opt-fuel").textContent = fmt(qiga.fuel_l, 0) + " L";
    const fuelSaving = greedy.fuel_l > 0 ? (1 - qiga.fuel_l / greedy.fuel_l) * 100 : 0;
    const co2Saving = greedy.emissions_t > 0 ? (1 - qiga.emissions_t / greedy.emissions_t) * 100 : 0;
    $("kpi-fuel-saving").textContent = fmt(fuelSaving, 1) + "%";
    $("kpi-co2-saving").textContent = fmt(co2Saving, 1) + "%";

    makeChart("chart-overview-compare", {
      type: "bar",
      data: {
        labels: ["Fuel (L)", "Cost (₹ '000)", "CO₂e (t)"],
        datasets: [
          { label: "Status quo (greedy)", data: [greedy.fuel_l, greedy.cost / 1000, greedy.emissions_t], backgroundColor: "#64748b" },
          { label: "Quantum-inspired optimized", data: [qiga.fuel_l, qiga.cost / 1000, qiga.emissions_t], backgroundColor: "#2dd4bf" },
        ],
      },
      options: { scales: { y: { beginAtZero: true } } },
    });
    toast("Live comparison complete — see Benchmarking tab for full detail.");
  } catch (e) {
    toast("Comparison failed: " + e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Live Demo Comparison";
  }
});

// ---------- Scenario Analysis ----------
$("btn-scenario").addEventListener("click", async () => {
  const btn = $("btn-scenario");
  btn.disabled = true;
  btn.textContent = "Running scenarios…";
  try {
    const r = await api("/api/scenario-analysis", "POST", currentScenario());
    $("scenario-results").classList.remove("hidden");
    const rows = r.scenarios.map(s => ({
      scenario: s.scenario,
      "fuel (L)": fmt(s.fuel_l, 0),
      cost: "₹" + fmt(s.cost, 0),
      "CO2e (t)": fmt(s.emissions_t, 2),
      "cargo served (t)": fmt(s.cargo_served_t, 0),
      feasible: s.feasible ? "✅" : "⚠️",
    }));
    fillTable("scenario-table", ["scenario", "fuel (L)", "cost", "CO2e (t)", "cargo served (t)", "feasible"], rows);

    const labels = r.scenarios.map(s => s.scenario);
    const barConfig = (data, color, label) => ({
      type: "bar",
      data: { labels, datasets: [{ label, data, backgroundColor: color }] },
      options: { plugins: { legend: { display: false } }, scales: { x: { ticks: { autoSkip: false, maxRotation: 30, minRotation: 20 } } } },
    });
    makeChart("chart-scenario-fuel", barConfig(r.scenarios.map(s => s.fuel_l), "#2dd4bf", "Fuel (L)"));
    makeChart("chart-scenario-cost", barConfig(r.scenarios.map(s => s.cost), "#fbbf24", "Cost"));
    makeChart("chart-scenario-co2", barConfig(r.scenarios.map(s => s.emissions_t), "#34d399", "CO2e (t)"));
    toast("Scenario analysis complete.");
  } catch (e) {
    toast("Scenario analysis failed: " + e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Scenario Analysis";
  }
});

// ---------- Benchmarking ----------
$("btn-benchmark").addEventListener("click", async () => {
  const btn = $("btn-benchmark");
  btn.disabled = true;
  btn.textContent = "Benchmarking…";
  try {
    const r = await api("/api/benchmark", "POST", currentScenario());
    $("benchmark-results").classList.remove("hidden");
    const rows = r.results.map(m => ({
      method: m.method,
      fitness: m.fitness,
      "fuel (L)": fmt(m.fuel_l, 0),
      "CO2e (t)": fmt(m.emissions_t, 2),
      "cargo served (t)": fmt(m.cargo_served_t, 0),
      feasible: m.feasible ? "✅" : "⚠️",
      "runtime (s)": m.runtime_seconds,
    }));
    fillTable("benchmark-table", ["method", "fitness", "fuel (L)", "CO2e (t)", "cargo served (t)", "feasible", "runtime (s)"], rows);

    const labels = r.results.map(m => m.method);
    makeChart("chart-bench-fitness", {
      type: "bar",
      data: { labels, datasets: [{ label: "Fitness", data: r.results.map(m => m.fitness), backgroundColor: ["#2dd4bf", "#60a5fa", "#fbbf24", "#fb7185"] }] },
      options: { plugins: { legend: { display: false } } },
    });
    makeChart("chart-bench-time", {
      type: "bar",
      data: { labels, datasets: [{ label: "Runtime (s)", data: r.results.map(m => m.runtime_seconds), backgroundColor: ["#2dd4bf", "#60a5fa", "#fbbf24", "#fb7185"] }] },
      options: { plugins: { legend: { display: false } } },
    });
    makeChart("chart-bench-convergence", {
      type: "line",
      data: {
        labels: r.results[0].convergence_history.map((_, i) => i + 1),
        datasets: r.results.filter(m => m.convergence_history.length > 1).map((m, i) => ({
          label: m.method,
          data: m.convergence_history,
          borderColor: ["#2dd4bf", "#60a5fa", "#fbbf24"][i] || "#fb7185",
          fill: false, tension: 0.2, pointRadius: 0,
        })),
      },
      options: { scales: { x: { title: { display: true, text: "Iteration" } }, y: { title: { display: true, text: "Best fitness so far" } } } },
    });
    toast("Benchmark complete.");
  } catch (e) {
    toast("Benchmark failed: " + e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Benchmark";
  }
});
