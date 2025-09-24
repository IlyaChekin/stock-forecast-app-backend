// build: 2025-09-24T18:30Z
const form = document.getElementById("upload-form");
const modelSelect = document.getElementById("model-select");
const lstmBox = document.getElementById("lstm-box");
const chartDiv = document.getElementById("chart");
const metricsDiv = document.getElementById("metrics");
const errorDiv = document.getElementById("error");

// helpers
const asNum = (x) => (x === null || x === undefined ? NaN : Number(x));
const num = (x, d = 4) => (Number.isFinite(asNum(x)) ? asNum(x).toFixed(d) : "—");
const pct = (x, d = 1) => (Number.isFinite(asNum(x)) ? asNum(x).toFixed(d) + "%" : "—");
const get = (obj, path, def = undefined) =>
  path.reduce((o, k) => (o && o[k] !== undefined ? o[k] : undefined), obj) ?? def;

function syncLstmBox() {
  lstmBox.style.display = modelSelect.value === "lstm" ? "block" : "none";
}
modelSelect.addEventListener("change", syncLstmBox);
syncLstmBox();

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorDiv.textContent = "";
  metricsDiv.textContent = "";
  chartDiv.innerHTML = "";

  const formData = new FormData(form);
  const file = formData.get("file");
  if (!(file && file.name && file.name.endsWith(".csv"))) {
    errorDiv.textContent = "Пожалуйста, загрузите CSV.";
    return;
  }

  try {
    const res = await fetch("/api/predict/regression", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();

    const target_col = data.target_col ?? "target";
    const n_lags = data.n_lags ?? 10;
    const test_size = data.test_size ?? 0.2;
    const model_name = data.model_name ?? data.model_key ?? "Model";

    const mTrain = get(data, ["metrics", "train"], {});
    const mTF = get(data, ["metrics", "teacher_forced"], {});
    const mREC = get(data, ["metrics", "recursive"], {});
    const mNTF = get(data, ["metrics", "naive_teacher_forced"], {});
    const mNRC = get(data, ["metrics", "naive_recursive"], {});
    const impTF = get(data, ["metrics", "improve_tf_vs_naive_tf_pct"], null);
    const impREC = get(data, ["metrics", "improve_rec_vs_naive_rec_pct"], null);

    metricsDiv.innerHTML = `
      <b>Модель:</b> ${model_name} (лаги=${num(n_lags, 0)}, test=${num(asNum(test_size) * 100, 0)}%)<br/>
      <b>Target:</b> ${target_col}<br/>
      <hr/>
      <b>TRAIN</b> → RMSE: ${num(mTrain.rmse)} | MAE: ${num(mTrain.mae)} | MAPE: ${num(mTrain.mape, 2)}%<br/>
      <b>TEST TF</b> → RMSE: ${num(mTF.rmse)} | MAE: ${num(mTF.mae)} | MAPE: ${num(mTF.mape, 2)}%<br/>
      <b>TEST REC</b> → RMSE: ${num(mREC.rmse)} | MAE: ${num(mREC.mae)} | MAPE: ${num(mREC.mape, 2)}%<br/>
      <b>Naive TF</b> → RMSE: ${num(mNTF.rmse)} | <b>Improve</b>: ${pct(impTF)}<br/>
      <b>Naive REC</b> → RMSE: ${num(mNRC.rmse)} | <b>Improve</b>: ${pct(impREC)}
    `;

    const tsTrain = get(data, ["timestamps", "train"], []);
    const tsTest = get(data, ["timestamps", "test"], []);
    const yTrain = get(data, ["series", "y_train"], []);
    const yTrainPred = get(data, ["series", "y_train_pred"], []);
    const yTest = get(data, ["series", "y_test"], []);
    const yPredTF = get(data, ["series", "y_pred_teacher_forced"], []);
    const yPredREC = get(data, ["series", "y_pred_recursive"], []);

    const traces = [
      { x: tsTrain, y: yTrain, mode: "lines", name: "Train • Факт" },
      { x: tsTrain, y: yTrainPred, mode: "lines", name: "Train • Прогноз (in-sample)", line: { dash: "dot" } },
      { x: tsTest, y: yTest, mode: "lines", name: "Test • Факт" },
      { x: tsTest, y: yPredTF, mode: "lines", name: "Test • Прогноз TF" },
      { x: tsTest, y: yPredREC, mode: "lines", name: "Test • Прогноз REC", line: { dash: "dash" } },
    ];

    Plotly.newPlot(chartDiv, traces, {
      title: "Train/Test • Факт vs Прогноз (TF/REC)",
      xaxis: { title: "Время" },
      yaxis: { title: target_col },
      margin: { t: 40, r: 20, b: 40, l: 50 },
      legend: { orientation: "h" },
    });
  } catch (err) {
    errorDiv.textContent = `Ошибка: ${err.message}`;
    console.error(err);
  }
});
