import { useInspection } from "../context/InspectionContext.jsx";

export default function DriftMonitoring() {
  const { history } = useInspection();
  const driftEvents = history.filter((item) => item.drift_detected);

  return (
    <>
      <div>
        <div className="topbar-title">Drift Monitoring</div>
        <div className="stat-foot">Confidence drift and stability overview</div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>Drift Summary</h3>
          <div className="stat-value">{driftEvents.length}</div>
          <div className="stat-foot">Total drift events</div>
        </div>
        <div className="card">
          <h3>Latest Drift Event</h3>
          {driftEvents.length ? (
            <div>
              <div style={{ fontWeight: 600 }}>{driftEvents[driftEvents.length - 1].defect_class}</div>
              <div className="stat-foot">{driftEvents[driftEvents.length - 1].timestamp}</div>
              <div className="stat-foot">
                Confidence: {Math.round(driftEvents[driftEvents.length - 1].confidence * 100)}%
              </div>
            </div>
          ) : (
            <div className="stat-foot">No drift events logged.</div>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Drift Event Log</h3>
        <div className="inspection-list">
          {driftEvents.length ? (
            driftEvents
              .slice()
              .reverse()
              .map((event) => (
                <div key={event.inspection_id} className="inspection-item">
                  <div>
                    <div style={{ fontWeight: 600 }}>{event.defect_class}</div>
                    <div className="stat-foot">{event.timestamp}</div>
                  </div>
                  <div className="stat-foot">{Math.round(event.confidence * 100)}%</div>
                </div>
              ))
          ) : (
            <div className="stat-foot">Awaiting first drift event.</div>
          )}
        </div>
      </div>
    </>
  );
}
