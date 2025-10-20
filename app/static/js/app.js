const fmtPct = (x) => (x*100).toFixed(1) + "%";

let dChart = null, hChart = null;

function renderDonut(ctxId, prob) {
  const el = document.getElementById(ctxId);
  const data = [prob, 1-prob];
  const cfg = {
    type: 'doughnut',
    data: {
      labels: ['Risk', ''],
      datasets: [{ data, borderWidth: 0 }]
    },
    options: {
      cutout: '70%',
      plugins: { legend: { display:false } },
      responsive: true,
      maintainAspectRatio: false
    }
  };
  const chart = new Chart(el, cfg);
  return chart;
}

function updateBadge(elId, decision) {
  const el = document.getElementById(elId);
  if (decision === 1) {
    el.className = "badge text-bg-danger";
    el.innerText = "High";
  } else {
    el.className = "badge text-bg-success";
    el.innerText = "Low";
  }
}

function renderFactors(elId, items) {
  const el = document.getElementById(elId);
  el.innerHTML = "";
  if (!items || items.length === 0) {
    el.innerHTML = '<span class="text-muted small">Install SHAP for factor insights.</span>';
    return;
  }
  items.slice(0,5).forEach(([name, val]) => {
    const badge = document.createElement("span");
    badge.className = "badge rounded-pill text-bg-light factor-badge";
    badge.innerText = name.replace("extra__", "");
    el.appendChild(badge);
  });
}

async function submitForm(e) {
  e.preventDefault();
  const form = document.getElementById("predict-form");
  const fd = new FormData(form);
  const payload = {};
  fd.forEach((v,k) => { if (v !== "") payload[k] = isNaN(v) ? v : Number(v); });

  const resp = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type":"application/json" },
    body: JSON.stringify(payload)
  });

  const resultsWrap = document.getElementById("results");
  const emptyState = document.getElementById("empty-state");
  const disclaimer = document.getElementById("disclaimer");

  if (!resp.ok) {
    const err = await resp.json().catch(()=>({error:"Unknown error"}));
    emptyState.innerText = "Error: " + (err.error || resp.statusText);
    resultsWrap.classList.add("d-none");
    emptyState.classList.remove("d-none");
    return;
  }

  const data = await resp.json();

  // Numbers
  const dProb = data.diabetes.prob, dThr = data.diabetes.threshold, dDec = data.diabetes.decision;
  const hProb = data.hypertension.prob, hThr = data.hypertension.threshold, hDec = data.hypertension.decision;

  // Update donuts
  if (dChart) dChart.destroy();
  if (hChart) hChart.destroy();
  dChart = renderDonut("d-chart", dProb);
  hChart = renderDonut("h-chart", hProb);

  // Badges, text, factors
  updateBadge("d-decision", dDec);
  updateBadge("h-decision", hDec);
  document.getElementById("d-prob").innerText = fmtPct(dProb);
  document.getElementById("d-thr").innerText  = fmtPct(dThr);
  document.getElementById("h-prob").innerText = fmtPct(hProb);
  document.getElementById("h-thr").innerText  = fmtPct(hThr);

  renderFactors("d-factors", data.diabetes.top_factors);
  renderFactors("h-factors", data.hypertension.top_factors);

  // Show disclaimer
  disclaimer.innerText = data.disclaimer || "Educational decision support.";

  // Toggle visibility
  resultsWrap.classList.remove("d-none");
  emptyState.classList.add("d-none");
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  form.addEventListener("submit", submitForm);
});
