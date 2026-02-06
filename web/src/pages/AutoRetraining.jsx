import { useInspection } from "../context/InspectionContext.jsx";

export default function AutoRetraining() {
  const { history } = useInspection();
  const retrainCandidates = history.filter((item) => item.drift_detected);

  return (
    <>
      <div>
        <div className="topbar-title">Auto-Retraining</div>
        <div className="stat-foot">Automated retraining readiness and triggers</div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>Retrain Queue</h3>
          <div className="stat-value">{retrainCandidates.length}</div>
          <div className="stat-foot">Drift events pending retrain</div>
        </div>
        <div className="card">
          <h3>Latest Trigger</h3>
          {retrainCandidates.length ? (
            <div>
              <div style={{ fontWeight: 600 }}>
                {retrainCandidates[retrainCandidates.length - 1].defect_class}
              </div>
              <div className="stat-foot">
                {retrainCandidates[retrainCandidates.length - 1].timestamp}
              </div>
            </div>
          ) : (
            <div className="stat-foot">No retraining triggers yet.</div>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Suggested Action</h3>
        <div className="stat-foot">
          After synthetic data generation, validate new samples and schedule a retraining run.
        </div>
        <div style={{ marginTop: 12 }}>
          <button className="button primary">Schedule Retraining</button>
        </div>
      </div>
    </>
  );
}
