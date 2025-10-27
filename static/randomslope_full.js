console.log("✅ randomslope_full.js loaded");

async function fetchRandomSlopeFull() {
  const params = {
    n: parseInt(document.getElementById("n_points").value),
    tau00: parseFloat(document.getElementById("tau00").value),
    tau11: parseFloat(document.getElementById("tau11").value),
    tau01: parseFloat(document.getElementById("tau01").value),
    sigma2: parseFloat(document.getElementById("sigma2").value)
  };

  const res = await fetch("/simulate_randomslope_full", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(params)
  });

  const data = await res.json();
  drawRandomSlopeCharts(data);
}

function drawRandomSlopeCharts(data) {
  const x = data.x.map(v => parseFloat(v.toFixed(2))); // clean x values

  // Destroy old chart if it exists
  if (window.varChart) window.varChart.destroy();

  const ctxVar = document.getElementById("varCanvas").getContext("2d");
  window.varChart = new Chart(ctxVar, {
    type: "line",
    data: {
      labels: x,
      datasets: [{
        label: "Diagonal of V (Var[y|x])",
        data: data.diagV,
        borderColor: "#2563eb",
        backgroundColor: "rgba(37,99,235,0.1)",
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 0
      }]
    },
    options: {
      plugins: {
        legend: { display: true, position: "top" }
      },
      scales: {
        x: {
          title: { display: true, text: "x (Covariate)" },
          ticks: { maxTicksLimit: 6 }
        },
        y: {
          title: { display: true, text: "Marginal Variance" },
          grid: { color: "rgba(0,0,0,0.05)" }
        }
      }
    }
  });

  // Draw heatmap with Plotly
  drawPlotlyHeatmap(x, data.corrV);
}

function drawPlotlyHeatmap(x, corrMatrix) {
  const trace = {
    z: corrMatrix,
    x: x.map(v => v.toFixed(2)),
    y: x.map(v => v.toFixed(2)),
    type: "heatmap",
    colorscale: "RdBu",
    zmin: -1,
    zmax: 1,
    reversescale: true,
    colorbar: {
      title: "Correlation",
      titleside: "right"
    }
  };

  const layout = {
    title: "Correlation Structure of V",
    xaxis: { title: "x_i", showgrid: false },
    yaxis: { title: "x_j", autorange: "reversed", showgrid: false },
    margin: { l: 60, r: 40, t: 40, b: 60 },
    width: 500,
    height: 500
  };

  Plotly.newPlot("covHeatmap", [trace], layout, {displayModeBar: false});
}
