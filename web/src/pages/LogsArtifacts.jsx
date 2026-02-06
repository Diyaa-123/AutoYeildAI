import { useInspection } from "../context/InspectionContext.jsx";

export default function LogsArtifacts() {
  const { inspection, history, metrics } = useInspection();

  return (
    <>
      <div>
        <div className="topbar-title">Logs & Artifacts</div>
        <div className="stat-foot">Latest inspection payloads and model metrics</div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>Latest Inspection</h3>
          <pre style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>
            {inspection ? JSON.stringify(inspection, null, 2) : "No inspection data yet."}
          </pre>
        </div>
        <div className="card">
          <h3>Model Metrics</h3>
          <pre style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>
            {metrics?.model_metrics ? JSON.stringify(metrics.model_metrics, null, 2) : "No model metrics found."}
          </pre>
        </div>
      </div>

      <div className="card">
        <h3>Inspection History</h3>
        <pre style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>
          {history.length ? JSON.stringify(history.slice(-10).reverse(), null, 2) : "History is empty."}
        </pre>
      </div>
    </>
  );
}
