
import { useEffect, useState } from "react";
import { checkHealth, getModelInfo, predictChurn } from "./services/api";
import "./App.css";

const initialCustomer = {
  gender: "Male",
  SeniorCitizen: 0,
  Partner: "Yes",
  Dependents: "No",
  tenure: 12,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "Yes",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 70.35,
  TotalCharges: 844.2,
};

const fields = [
  { name: "gender", label: "Gender", options: ["Male", "Female"] },
  { name: "SeniorCitizen", label: "Senior citizen", options: ["0", "1"] },
  { name: "Partner", label: "Partner", options: ["Yes", "No"] },
  { name: "Dependents", label: "Dependents", options: ["Yes", "No"] },
  { name: "tenure", label: "Tenure (months)", type: "number", min: 0, max: 100 },
  { name: "PhoneService", label: "Phone service", options: ["Yes", "No"] },
  {
    name: "MultipleLines",
    label: "Multiple lines",
    options: ["Yes", "No", "No phone service"],
  },
  {
    name: "InternetService",
    label: "Internet service",
    options: ["DSL", "Fiber optic", "No"],
  },
  {
    name: "OnlineSecurity",
    label: "Online security",
    options: ["Yes", "No", "No internet service"],
  },
  {
    name: "OnlineBackup",
    label: "Online backup",
    options: ["Yes", "No", "No internet service"],
  },
  {
    name: "DeviceProtection",
    label: "Device protection",
    options: ["Yes", "No", "No internet service"],
  },
  {
    name: "TechSupport",
    label: "Tech support",
    options: ["Yes", "No", "No internet service"],
  },
  {
    name: "StreamingTV",
    label: "Streaming TV",
    options: ["Yes", "No", "No internet service"],
  },
  {
    name: "StreamingMovies",
    label: "Streaming movies",
    options: ["Yes", "No", "No internet service"],
  },
  {
    name: "Contract",
    label: "Contract",
    options: ["Month-to-month", "One year", "Two year"],
  },
  {
    name: "PaperlessBilling",
    label: "Paperless billing",
    options: ["Yes", "No"],
  },
  {
    name: "PaymentMethod",
    label: "Payment method",
    options: [
      "Electronic check",
      "Mailed check",
      "Bank transfer (automatic)",
      "Credit card (automatic)",
    ],
  },
  {
    name: "MonthlyCharges",
    label: "Monthly charges",
    type: "number",
    min: 0,
    step: "0.01",
  },
  {
    name: "TotalCharges",
    label: "Total charges",
    type: "number",
    min: 0,
    step: "0.01",
  },
];

function App() {
  const [customer, setCustomer] = useState(initialCustomer);
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function loadStatus() {
      try {
        const [healthData, modelData] = await Promise.all([
          checkHealth(),
          getModelInfo(),
        ]);
        setHealth(healthData);
        setModelInfo(modelData);
      } catch {
        setHealth(null);
      }
    }

    loadStatus();
  }, []);

  function handleChange(event) {
    const { name, value, type } = event.target;

    setCustomer((previous) => ({
      ...previous,
      [name]:
        type === "number"
          ? value === ""
            ? ""
            : Number(value)
          : name === "SeniorCitizen"
            ? Number(value)
            : value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await predictChurn(customer);
      setResult(data);
    } catch (err) {
      setError(err.message || "Prediction failed. Check the backend.");
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    setCustomer(initialCustomer);
    setResult(null);
    setError("");
  }

  const probability = result
    ? (result.churn_probability * 100).toFixed(1)
    : null;

  return (
    <main className="dashboard">
      <header className="topbar">
        <div>
          <p className="eyebrow">MACHINE LEARNING PROJECT</p>
          <h1>Customer Churn <span>Intelligence</span></h1>
          <p className="subtitle">
            Predict customer retention risk using your trained ML model.
          </p>
        </div>

        <div className={`connection ${health ? "online" : "offline"}`}>
          <span className="status-dot" />
          {health ? "API Connected" : "API Disconnected"}
        </div>
      </header>

      <section className="summary-grid">
        <article className="summary-card">
          <span className="summary-icon">◉</span>
          <p>Backend status</p>
          <h2>{health ? "Healthy" : "Unavailable"}</h2>
          <small>FastAPI service</small>
        </article>

        <article className="summary-card">
          <span className="summary-icon">◇</span>
          <p>Model version</p>
          <h2>{modelInfo?.model_version || "—"}</h2>
          <small>{modelInfo?.model_type || "Waiting for model"}</small>
        </article>

        <article className="summary-card">
          <span className="summary-icon">◎</span>
          <p>Latest prediction</p>
          <h2>{result ? result.prediction === "Yes" ? "Churn risk" : "Retained" : "—"}</h2>
          <small>{result ? `${probability}% churn probability` : "Submit customer details"}</small>
        </article>
      </section>

      <div className="main-grid">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">CUSTOMER PROFILE</p>
              <h2>Predict churn</h2>
              <p className="muted">Enter the customer's account details.</p>
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              {fields.map((field) => (
                <label className="field" key={field.name}>
                  <span>{field.label}</span>

                  {field.options ? (
                    <select
                      name={field.name}
                      value={customer[field.name]}
                      onChange={handleChange}
                    >
                      {field.options.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      name={field.name}
                      type={field.type || "number"}
                      min={field.min}
                      max={field.max}
                      step={field.step || "1"}
                      value={customer[field.name]}
                      onChange={handleChange}
                      required
                    />
                  )}
                </label>
              ))}
            </div>

            <div className="form-actions">
              <button className="button-secondary" type="button" onClick={resetForm}>
                Reset
              </button>
              <button
                className="button-primary"
                type="submit"
                disabled={loading || !health}
              >
                {loading ? "Analyzing..." : "Predict churn →"}
              </button>
            </div>
          </form>

          {!health && (
            <p className="notice">
              Start FastAPI and check the connection before predicting.
            </p>
          )}
        </section>

        <aside className="panel result-panel">
          <p className="eyebrow">MODEL OUTPUT</p>
          <h2>Prediction result</h2>

          {!result && !error && (
            <div className="empty-state">
              <div className="empty-icon">✳</div>
              <h3>Ready to analyze</h3>
              <p>Submit customer details to see the model's churn prediction.</p>
            </div>
          )}

          {result && (
            <div className="result-content">
              <div className={`result-badge ${result.prediction === "Yes" ? "risk" : "safe"}`}>
                {result.prediction === "Yes" ? "High churn signal" : "No churn predicted"}
              </div>

              <div className="probability">
                <strong>{probability}%</strong>
                <span>Churn probability</span>
              </div>

              <div className="progress-track">
                <div
                  className={`progress-fill ${result.prediction === "Yes" ? "risk-fill" : "safe-fill"}`}
                  style={{ width: `${Math.min(100, Math.max(0, Number(probability)))}%` }}
                />
              </div>

              <div className="result-row">
                <span>Predicted churn</span>
                <strong>{result.prediction}</strong>
              </div>
              <div className="result-row">
                <span>Model version</span>
                <strong>{result.model_version}</strong>
              </div>
              <div className="database-note">
                <span>✓</span>
                {result.logged_to_database
                  ? "Prediction saved to PostgreSQL"
                  : "Database logging not confirmed"}
              </div>
              <p className="disclaimer">
                This is a model estimate, not a guarantee of customer behavior.
              </p>
            </div>
          )}

          {error && <div className="error-message">{error}</div>}
        </aside>
      </div>

      <footer>
        <span>Customer Churn ML</span>
        <span>React · FastAPI · PostgreSQL</span>
      </footer>
    </main>
  );
}

export default App;
