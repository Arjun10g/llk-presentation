// ============================================================
// Main Frontend Script for MLM Visualization App
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
  const navItems = document.querySelectorAll(".sidebar nav ul li");
  const sections = document.querySelectorAll(".section");

  navItems.forEach(item => {
    item.addEventListener("click", () => {
      // Remove active state from all
      navItems.forEach(li => li.classList.remove("active"));
      sections.forEach(sec => sec.classList.remove("active"));

      // Add active to clicked item + corresponding section
      item.classList.add("active");
      const target = item.getAttribute("data-section");
      document.getElementById(target).classList.add("active");

      // Smooth scroll for better UX
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });
  // ==========================================================
  // TAB 1: LIKELIHOOD SURFACE SIMULATION
  // ==========================================================
  const btn = document.getElementById("simulate");
  if (btn) {
    // --- create progress UI ---
    const progressBar = document.createElement("div");
    const progressText = document.createElement("div");
    progressBar.id = "progressBar";
    progressText.id = "progressText";
    document.querySelector(".controls").after(progressBar);
    document.querySelector(".controls").after(progressText);

    btn.addEventListener("click", async () => {
      btn.textContent = "Computing...";
      btn.disabled = true;
      progressBar.style.width = "0%";
      progressBar.classList.add("visible");
      progressText.textContent = "Starting computations...";

      const payload = {
        J: +document.getElementById("J").value,
        n_per: +document.getElementById("n_per").value,
        tauMax: +document.getElementById("tauMax").value,
        sigmaMax: +document.getElementById("sigmaMax").value,
        steps: 15
      };

      try {
        // --- fake progress animation while computing ---
        for (let i = 1; i <= 10; i++) {
          setTimeout(() => {
            progressBar.style.width = `${i * 10}%`;
            progressText.textContent = `Computing... ${i * 10}%`;
          }, i * 300);
        }

        // --- actual backend call ---
        const res = await fetch("/compute_surface", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Server error " + res.status);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        progressText.textContent = "Finished computing surface.";
        setTimeout(() => progressBar.classList.remove("visible"), 1000);

        // --- plots ---
        plotSurface3D(data);
        plotContour2D(data);
        plotProfiles(data);

        document.getElementById("plot3D").scrollIntoView({ behavior: "smooth" });

      } catch (err) {
        progressText.textContent = "❌ " + err.message;
        progressBar.classList.remove("visible");
        alert(err.message);
      } finally {
        btn.textContent = "Simulate & Compute";
        btn.disabled = false;
      }
    });
  }

  // ==========================================================
  // Plotly Visualization Functions for TAB 1
  // ==========================================================
  function plotSurface3D(data) {
    Plotly.purge("plot3D");

    const surface = {
      x: data.sigma,
      y: data.tau,
      z: data.logLik,
      type: "surface",
      colorscale: "Viridis",
      opacity: 0.95,
      name: "Log-Likelihood Surface"
    };

    const mlePoint = {
      x: [data.mle.sigma],
      y: [data.mle.tau],
      z: [data.mle.value],
      mode: "markers+text",
      text: [`MLE<br>τ²=${data.mle.tau.toFixed(3)}<br>σ²=${data.mle.sigma.toFixed(3)}`],
      textposition: "top center",
      marker: { color: "red", size: 7, symbol: "x" },
      type: "scatter3d",
      name: "MLE"
    };

    const layout = {
      title: { text: "3D Log-Likelihood Surface", font: { size: 18 } },
      scene: {
        xaxis: { title: "σ² (Residual Variance)" },
        yaxis: { title: "τ² (Random Intercept Variance)" },
        zaxis: { title: "Log-Likelihood" }
      },
      margin: { l: 0, r: 0, t: 40, b: 0 },
      height: 550
    };

    Plotly.newPlot("plot3D", [surface, mlePoint], layout);
  }

  function plotContour2D(data) {
    Plotly.purge("plot2D");

    const contour = {
      x: data.sigma,
      y: data.tau,
      z: data.logLik,
      type: "contour",
      colorscale: "Viridis",
      contours: { coloring: "heatmap" },
      name: "Likelihood"
    };

    const mleMarker = {
      x: [data.mle.sigma],
      y: [data.mle.tau],
      mode: "markers+text",
      text: [`MLE<br>τ²=${data.mle.tau.toFixed(3)}<br>σ²=${data.mle.sigma.toFixed(3)}`],
      textposition: "top center",
      marker: { color: "red", size: 12, symbol: "x" },
      name: "MLE"
    };

    const lines = [
      {
        type: "line",
        x0: data.mle.sigma,
        x1: data.mle.sigma,
        y0: Math.min(...data.tau),
        y1: Math.max(...data.tau),
        line: { color: "red", dash: "dash", width: 1 }
      },
      {
        type: "line",
        y0: data.mle.tau,
        y1: data.mle.tau,
        x0: Math.min(...data.sigma),
        x1: Math.max(...data.sigma),
        line: { color: "red", dash: "dash", width: 1 }
      }
    ];

    const layout = {
      title: "2D Contour of Log-Likelihood with MLE Crosshairs",
      xaxis: { title: "σ² (Residual Variance)" },
      yaxis: { title: "τ² (Random Intercept Variance)" },
      shapes: lines,
      margin: { l: 60, r: 30, t: 50, b: 60 }
    };

    Plotly.newPlot("plot2D", [contour, mleMarker], layout);
  }

  function plotProfiles(data) {
    // τ² profile
    Plotly.purge("tauProfile");
    const tauTrace = {
      x: data.tau,
      y: data.tauProfile,
      mode: "lines",
      line: { color: "steelblue", width: 2 },
      name: "τ² Profile"
    };
    const tauMLE = {
      x: [data.mle.tau],
      y: [Math.max(...data.tauProfile)],
      mode: "markers+text",
      text: [`MLE<br>τ²=${data.mle.tau.toFixed(3)}`],
      textposition: "top center",
      marker: { color: "red", size: 8, symbol: "x" },
      name: "MLE"
    };
    Plotly.newPlot("tauProfile", [tauTrace, tauMLE], {
      title: "Profile Likelihood (τ²)",
      xaxis: { title: "τ²" },
      yaxis: { title: "Log-Likelihood" },
      margin: { t: 50, l: 60, r: 30, b: 60 }
    });

    // σ² profile
    Plotly.purge("sigmaProfile");
    const sigmaTrace = {
      x: data.sigma,
      y: data.sigmaProfile,
      mode: "lines",
      line: { color: "seagreen", width: 2 },
      name: "σ² Profile"
    };
    const sigmaMLE = {
      x: [data.mle.sigma],
      y: [Math.max(...data.sigmaProfile)],
      mode: "markers+text",
      text: [`MLE<br>σ²=${data.mle.sigma.toFixed(3)}`],
      textposition: "top center",
      marker: { color: "red", size: 8, symbol: "x" },
      name: "MLE"
    };
    Plotly.newPlot("sigmaProfile", [sigmaTrace, sigmaMLE], {
      title: "Profile Likelihood (σ²)",
      xaxis: { title: "σ²" },
      yaxis: { title: "Log-Likelihood" },
      margin: { t: 50, l: 60, r: 30, b: 60 }
    });
  }

  // ==========================================================
  // TAB 2: MATRIX ANATOMY (RANDOM SLOPE / INTERCEPT)
  // ==========================================================
  const matrixBtn = document.getElementById("computeMatrix");
  if (matrixBtn) {
    matrixBtn.addEventListener("click", async () => {
      matrixBtn.textContent = "Computing...";
      matrixBtn.disabled = true;

      const payload = {
        J: +document.getElementById("J_mat").value,
        n_per: +document.getElementById("n_mat").value,
        structure: document.getElementById("structure_mat").value,
        tau2: +document.getElementById("tau_mat").value,
        tau2_slope: +document.getElementById("tau_slope_mat").value,
        rho: +document.getElementById("rho_mat").value,
        sigma2: +document.getElementById("sig_mat").value
      };

      try {
        const res = await fetch("/matrix_anatomy", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Server error " + res.status);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        plotMatrix("plotG", data.G, "G (Random Effect Covariance)");
        plotMatrix("plotR", data.R, "R (Residual Covariance)");
        plotMatrix("plotV", data.V, "V = ZGZ′ + R (Marginal Covariance)");
        plotMatrix("plotVinv", data.Vinv, "V⁻¹ (Precision / Information Matrix)");
        plotR2Decomp(data);
        showMatrixSummary(data);

        document.getElementById("plotG").scrollIntoView({ behavior: "smooth" });

      } catch (err) {
        alert("❌ " + err.message);
      } finally {
        matrixBtn.textContent = "Compute Matrices";
        matrixBtn.disabled = false;
      }
    });
  }

  function plotMatrix(divId, matrix, title) {
    Plotly.purge(divId);
    const trace = {
      z: matrix,
      type: "heatmap",
      colorscale: "RdBu",
      reversescale: true,
      showscale: true
    };
    const layout = {
      title: title,
      margin: { l: 40, r: 20, t: 50, b: 30 },
      xaxis: { showgrid: false, zeroline: false },
      yaxis: { showgrid: false, zeroline: false },
    };
    Plotly.newPlot(divId, [trace], layout);
  }

  function plotR2Decomp(data) {
    Plotly.purge("r2plot");
    const trace = {
      x: ["Between (τ²)", "Within (σ²)"],
      y: [data.components.between, data.components.within],
      type: "bar",
      marker: { color: ["#2563eb", "#16a34a"] }
    };
    const layout = {
      title: `Variance Partitioning (ICC ≈ ${data.icc.toFixed(3)})`,
      yaxis: { title: "Variance Component" },
      xaxis: { title: "" },
      margin: { t: 50, l: 60, r: 30, b: 60 }
    };
    Plotly.newPlot("r2plot", [trace], layout);
  }

  function showMatrixSummary(data) {
    const info = document.getElementById("matrixInfo");
    info.innerHTML = `
      <p><strong>Variance decomposition:</strong><br>
      Between (τ²) ≈ ${data.components.between.toFixed(3)}<br>
      Within (σ²) ≈ ${data.components.within.toFixed(3)}<br>
      Total ≈ ${data.components.total.toFixed(3)}</p>
      <p><strong>Intraclass Correlation (ICC):</strong> ${data.icc.toFixed(3)}</p>
      <p><strong>Interpretation:</strong><br>
      ICC quantifies how much of total variance is due to clustering.<br>
      High ICC ⇒ greater similarity within clusters ⇒ stronger hierarchical dependence.<br>
      V⁻¹ encodes <em>precision</em>: elements of V⁻¹ tell us how each observation<br>
      influences the global fit — sharp curvature → more information.</p>
    `;
  }


});
