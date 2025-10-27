// ============================================================
// Main Frontend Script for MLM Visualization App
// ============================================================

document.addEventListener("DOMContentLoaded", () => {

    // -----------------------------
    // FIXED vs RANDOM D3 block
    // -----------------------------
    (async function(){
      const svg = d3.select("#chart");
      if (svg.empty()) return;  // ✅ Skip D3 if chart not on this page
    
      const W = +svg.attr("width") || 600;
      const H = +svg.attr("height") || 400;
      const M = {t: 30, r: 30, b: 50, l: 60};
      const iw = W - M.l - M.r;
      const ih = H - M.t - M.b;
    
      const g = svg.append("g").attr("transform", `translate(${M.l},${M.t})`);
      const color = d3.scaleOrdinal(d3.schemeTableau10);
    
      try {
        const res = await fetch("/simulate_fxre_visual");
        if (!res.ok) throw new Error("Server error " + res.status);
        const data = await res.json();
        const allPts = data.groups.flatMap(d => d.points);
    
        const x = d3.scaleLinear().domain(d3.extent(allPts, d => d.x)).range([0, iw]);
        const y = d3.scaleLinear().domain(d3.extent(allPts, d => d.y)).nice().range([ih, 0]);
    
        g.append("g").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x));
        g.append("g").call(d3.axisLeft(y));
    
        // Dots
        const dots = g.selectAll(".dot")
          .data(allPts)
          .enter()
          .append("circle")
          .attr("class", "dot")
          .attr("r", 3)
          .attr("cx", d => x(d.x))
          .attr("cy", d => y(d.y))
          .attr("fill", "#222");
    
        // Global regression
        const lm = linearFit(allPts.map(d => [d.x, d.y]));
        const X = d3.extent(allPts, d => d.x);
        const globalLine = X.map(xx => ({x: xx, y: lm.b0 + lm.b1 * xx}));
        const lineGen = d3.line().x(d => x(d.x)).y(d => y(d.y));
        g.append("path")
          .datum(globalLine)
          .attr("class", "line main-line")
          .attr("stroke", "#222")
          .attr("d", lineGen);
    
        // Group lines
        const groupLines = g.selectAll(".group-line")
          .data(data.groups)
          .enter()
          .append("path")
          .attr("class", "line group-line")
          .attr("stroke", d => color(d.group))
          .attr("d", d => {
            const fit = linearFit(d.points.map(p => [p.x, p.y]));
            const xx = d3.extent(d.points, p => p.x);
            const yy = xx.map(xv => fit.b0 + fit.b1 * xv);
            return lineGen(xx.map((xv,i) => ({x: xv, y: yy[i]})));
          })
          .style("opacity", 0);
    
        // Toggle button
        const btn = d3.select("#toggle");
        if (!btn.empty()) {
          let revealIndex = 0;
          btn.on("click", () => {
            if (revealIndex === 0)
              g.select(".main-line").transition().duration(800).style("opacity", 0.4);
    
            if (revealIndex < data.groups.length) {
              const grp = data.groups[revealIndex];
              const c = color(grp.group);
              dots.filter(d => d.g === grp.group)
                .transition().duration(800)
                .attr("r", 4).attr("fill", c);
              g.selectAll(".group-line")
                .filter(d => d.group === grp.group)
                .transition().duration(800)
                .style("opacity", 1);
              revealIndex++;
              if (revealIndex === data.groups.length) btn.text("Reset");
            } else {
              revealIndex = 0;
              dots.transition().duration(800).attr("fill", "#222").attr("r", 3).style("opacity", 0.9);
              groupLines.transition().duration(800).style("opacity", 0);
              g.select(".main-line").transition().duration(800).style("opacity", 1);
              btn.text("Reveal Next Group");
            }
          });
        }
    
        // Helper for regression
        function linearFit(points){
          const n = points.length;
          const sumX = d3.sum(points, d => d[0]);
          const sumY = d3.sum(points, d => d[1]);
          const sumXY = d3.sum(points, d => d[0]*d[1]);
          const sumX2 = d3.sum(points, d => d[0]*d[0]);
          const b1 = (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX);
          const b0 = (sumY - b1*sumX) / n;
          return {b0, b1};
        }
    
      } catch (err) {
        console.error("D3 visual error:", err);
      }
    
    })();
    // -----------------------------
    // END FIXED vs RANDOM D3 block
    /**
 * MLMViz Frontend (client-side)
 * Connects Flask endpoints to D3 visualization.
 * Unique namespace: window.MLMVIZ_FRONT
 */
(() => {
  const NS = {};
  const svg = d3.select("#mlmviz-canvas");
  const width = 900;
  const height = 420;
  const margin = { top: 30, right: 40, bottom: 40, left: 50 };
  const X = d3.scaleLinear().domain([-4, 4]).range([margin.left, width - margin.right]);
  const Y = d3.scaleLinear().domain([-2.2, 3.6]).range([height - margin.bottom, margin.top]);

  // Layers
  const gCurve = svg.append("g").attr("id", "mlmviz-layer-curve");
  const gBall = svg.append("g").attr("id", "mlmviz-layer-ball");
  const gTrail = svg.append("g").attr("id", "mlmviz-layer-trail");

  const pathCurve = gCurve.append("path")
    .attr("id", "mlmviz-path-main")
    .attr("fill", "none")
    .attr("stroke", "#2563eb")
    .attr("stroke-width", 3);

  const ball = gBall.append("circle")
    .attr("id", "mlmviz-ball")
    .attr("r", 8)
    .attr("fill", "#ef4444")
    .attr("stroke", "#991b1b")
    .attr("stroke-width", 1.5);

  /** ===========================
   * 1. Utility Fetchers
   * =========================== */
  async function getCurve(mode = "quadratic", a = 1.0, angle = 30) {
    const url = `/mlmviz/curve?mode=${mode}&a=${a}&angle_deg=${angle}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Curve fetch failed: ${res.status}`);
    return res.json();
  }

  async function resetState() {
    const res = await fetch("/mlmviz/state/reset");
    if (!res.ok) throw new Error("Reset failed");
    return res.json();
  }

  async function stepState(n = 1, trace = false) {
    const res = await fetch("/mlmviz/state/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n, return_trace: trace })
    });
    if (!res.ok) throw new Error("Step failed");
    return res.json();
  }

  async function updateConfig(payload) {
    const res = await fetch("/mlmviz/state/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("Config update failed");
    return res.json();
  }

  /** ===========================
   * 2. Draw functions
   * =========================== */
  function drawCurve(data) {
    const pts = data.x.map((x, i) => [X(x), Y(data.y[i])]);
    const pathData = d3.line()(pts);
    pathCurve.attr("d", pathData);
  }

  function drawBall(x, y) {
    ball.attr("cx", X(x)).attr("cy", Y(y));
  }

  function drawTrail(x, y) {
    gTrail.append("circle")
      .attr("cx", X(x))
      .attr("cy", Y(y))
      .attr("r", 3)
      .attr("fill", "#ef4444")
      .attr("opacity", 0.15)
      .transition()
      .duration(800)
      .attr("opacity", 0)
      .remove();
  }

  /** ===========================
   * 3. Animation Loop
   * =========================== */
  let animating = false;

  async function animateDrop() {
    animating = true;
    let result = await stepState(1);
    let frame = 0;

    function frameLoop() {
      if (!animating) return;
      stepState(1, false).then(res => {
        const { state, at_rest } = res;
        drawBall(state.x, state.y);
        if (frame % 5 === 0) drawTrail(state.x, state.y);
        frame++;
        if (!at_rest) requestAnimationFrame(frameLoop);
        else animating = false;
      });
    }
    frameLoop();
  }

  /** ===========================
   * 4. Controls Wiring
   * =========================== */
  const curvatureSlider = document.getElementById("mlmviz-curvature-slider");
  const curvatureValue = document.getElementById("mlmviz-curvature-value");
  const angleSlider = document.getElementById("mlmviz-angle-slider");
  const angleValue = document.getElementById("mlmviz-angle-value");
  const boundaryToggle = document.getElementById("mlmviz-boundary-toggle");
  const resetButton = document.getElementById("mlmviz-reset");

  curvatureSlider.oninput = async () => {
    const a = parseFloat(curvatureSlider.value);
    curvatureValue.textContent = a.toFixed(2);
    await updateConfig({ a });
    const data = await getCurve(boundaryToggle.checked ? "boundary" : "quadratic", a, parseFloat(angleSlider.value));
    drawCurve(data);
  };

  angleSlider.oninput = async () => {
    const angle = parseFloat(angleSlider.value);
    angleValue.textContent = angle.toFixed(0);
    await updateConfig({ angle_deg: angle });
    if (boundaryToggle.checked) {
      const data = await getCurve("boundary", parseFloat(curvatureSlider.value), angle);
      drawCurve(data);
    }
  };

  boundaryToggle.onchange = async () => {
    const mode = boundaryToggle.checked ? "boundary" : "quadratic";
    await updateConfig({ mode });
    const data = await getCurve(mode, parseFloat(curvatureSlider.value), parseFloat(angleSlider.value));
    drawCurve(data);
  };

  resetButton.onclick = async () => {
    animating = false;
    const state = await resetState();
    const data = await getCurve(state.config.mode, state.config.a, state.config.angle_deg);
    drawCurve(data);
    drawBall(state.state.x, state.state.y);
    animateDrop();
  };

  /** ===========================
   * 5. Initial Load
   * =========================== */
  (async function init() {
    const initial = await resetState();
    const data = await getCurve(initial.config.mode, initial.config.a, initial.config.angle_deg);
    drawCurve(data);
    drawBall(initial.state.x, initial.state.y);
    animateDrop();
  })();
})();
async function fetchMLREML() {
    const payload = {
      J: +document.getElementById("viz_J").value,
      n_per: +document.getElementById("viz_n").value,
      tau2_int: +document.getElementById("viz_tau00").value,
      tau2_slope: +document.getElementById("viz_tau11").value,
      rho: +document.getElementById("viz_rho").value,
      sigma2: +document.getElementById("viz_sigma2").value,
      beta0: +document.getElementById("viz_b0").value,
      beta1: +document.getElementById("viz_b1").value,
      random_slope: document.getElementById("viz_rslope").checked
    };
  
    const res = await fetch("/mlreml_quick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
  
    if (!res.ok) throw new Error((await res.json()).error || "Request failed");
    return res.json();
  }
  
  function drawBars(data) {
    const svg = d3.select("#viz_bars");
    svg.selectAll("*").remove();
  
    const slope = data.inputs.random_slope;
    const w = +svg.attr("width"), h = +svg.attr("height");
    const m = {t: 30, r: 30, b: 50, l: 60};
    const IW = w - m.l - m.r, IH = h - m.t - m.b;
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
  
    // parameter keys in backend JSON
    const params = slope
      ? ["tau2_intercept", "tau2_slope", "sigma2"]
      : ["tau2_intercept", "sigma2"];
  
    const labels = {
      tau2_intercept: "τ₀₀ (Intercept Var)",
      tau2_slope: "τ₁₁ (Slope Var)",
      sigma2: "σ² (Residual Var)"
    };
  
    const series = ["True", "ML", "REML"];
  
    const dataset = params.map(p => ({
      param: p,
      True: data.inputs[p + "_true"],
      ML: data.ml.var[p],
      REML: data.reml.var[p]
    }));
  
    const x0 = d3.scaleBand().domain(params).range([0, IW]).padding(0.25);
    const x1 = d3.scaleBand().domain(series).range([0, x0.bandwidth()]).padding(0.12);
    const maxVal = d3.max(dataset.flatMap(d => [d.True, d.ML, d.REML])) || 1;
    const y = d3.scaleLinear().domain([0, maxVal * 1.15]).nice().range([IH, 0]);
    const color = d3.scaleOrdinal().domain(series).range(["#10b981", "#2563eb", "#f97316"]);
  
    // axes
    g.append("g")
      .attr("transform", `translate(0,${IH})`)
      .call(d3.axisBottom(x0).tickFormat(d => labels[d] || d));
    g.append("g").call(d3.axisLeft(y));
  
    // bars
    const groups = g.selectAll(".grp")
      .data(dataset)
      .enter().append("g")
      .attr("class", "grp")
      .attr("transform", d => `translate(${x0(d.param)},0)`);
  
    groups.selectAll("rect")
      .data(d => series.map(s => ({key: s, val: d[s]})))
      .enter().append("rect")
      .attr("x", d => x1(d.key))
      .attr("width", x1.bandwidth())
      .attr("y", IH)
      .attr("height", 0)
      .attr("fill", d => color(d.key))
      .transition().duration(400)
      .attr("y", d => y(d.val))
      .attr("height", d => IH - y(d.val));
  
    // value labels
    groups.selectAll("text.val")
      .data(d => series.map(s => ({key: s, val: d[s]})))
      .enter().append("text")
      .attr("class", "val")
      .attr("x", d => x1(d.key) + x1.bandwidth()/2)
      .attr("y", d => y(d.val) - 5)
      .attr("text-anchor", "middle")
      .attr("font-size", "11px")
      .attr("fill", "#334155")
      .text(d => d.val != null ? d.val.toFixed(3) : "NA");
  
    // legend
    const legend = g.append("g").attr("transform", "translate(0,-20)");
    series.forEach((s, i) => {
      const L = legend.append("g").attr("transform", `translate(${i * 120},0)`);
      L.append("rect").attr("width", 12).attr("height", 12).attr("rx", 2).attr("fill", color(s));
      L.append("text").attr("x", 18).attr("y", 10).attr("font-size", "12px").text(s);
    });
  
    g.append("text")
      .attr("x", IW / 2)
      .attr("y", -10)
      .attr("text-anchor", "middle")
      .attr("font-size", "13px")
      .attr("font-family", "monospace")
      .text("Variance Components: True vs ML vs REML");
  }
  
  
  function drawTable(data) {
    const tbody = document.getElementById("viz_tbody");
    tbody.innerHTML = "";
  
    const terms = new Set([
      ...Object.keys(data.ml.fixed.coef),
      ...Object.keys(data.reml.fixed.coef)
    ]);
    const trueVals = {"Intercept": data.inputs.beta0_true, "x": data.inputs.beta1_true};
  
    for (const term of terms) {
      const tr = document.createElement("tr");
      const addCell = (txt) => {
        const td = document.createElement("td");
        td.style.padding = "6px";
        td.style.borderBottom = "1px solid #e2e8f0";
        td.textContent = txt;
        tr.appendChild(td);
      };
  
      addCell(term);
      addCell(trueVals[term] != null ? trueVals[term].toFixed(3) : "—");
  
      const mlc = data.ml.fixed.coef[term], mls = data.ml.fixed.se[term];
      addCell(mlc != null ? `${mlc.toFixed(3)} ± ${mls.toFixed(3)}` : "—");
  
      const rec = data.reml.fixed.coef[term], res = data.reml.fixed.se[term];
      addCell(rec != null ? `${rec.toFixed(3)} ± ${res.toFixed(3)}` : "—");
  
      tbody.appendChild(tr);
    }
  }
  
  async function runViz() {
    const err = document.getElementById("viz_error");
    err.textContent = "";
    try {
      const data = await fetchMLREML();
      drawBars(data);
      drawTable(data);
    } catch (e) {
      err.textContent = e.message || String(e);
    }
  }
  
  document.getElementById("viz_run").addEventListener("click", runViz);
  
}
);