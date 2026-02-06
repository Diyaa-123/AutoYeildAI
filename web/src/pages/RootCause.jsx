import { useInspection } from "../context/InspectionContext.jsx";

export default function RootCause() {
  const { inspection } = useInspection();
  const hasData = Boolean(inspection);
  const reasoning = inspection?.reasoning || {};
  const summary = reasoning.summary || reasoning.cause_summary || "";

  return (
    <>
      <div>
        <div className="topbar-title">Root Cause Analysis</div>
        <div className="stat-foot">LLM-powered insights</div>
      </div>

      <div className="card">
        <h3>Structured Evidence Input</h3>
        <div className="grid-4">
          <div>
            <div className="stat-foot">Defect Type</div>
            <div style={{ fontWeight: 600 }}>
              {hasData ? inspection.defect_class : "Scratch"}
            </div>
          </div>
          <div>
            <div className="stat-foot">Confidence</div>
            <div style={{ fontWeight: 600 }}>
              {hasData ? `${Math.round(inspection.confidence * 100)}%` : "96.5%"}
            </div>
          </div>
          <div>
            <div className="stat-foot">Location</div>
            <div style={{ fontWeight: 600 }}>Quadrant NE</div>
          </div>
          <div>
            <div className="stat-foot">Trend</div>
            <div style={{ fontWeight: 600 }}>Increasing</div>
          </div>
        </div>
        <div className="stat-foot" style={{ marginTop: 10 }}>
          Note: Only structured evidence is passed to the LLM. Raw images are never sent.
        </div>
      </div>

      <div className="card" style={{ borderColor: "rgba(245, 158, 11, 0.4)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontWeight: 600 }}>Risk Assessment</div>
            <div className="stat-foot">
              {reasoning.severity_assessment
                ? `${reasoning.severity_assessment} risk based on current pattern.`
                : "2-5% yield loss if uncorrected"}
            </div>
          </div>
          <div className="chip warn">
            {reasoning.severity_assessment ? `${reasoning.severity_assessment} Risk` : "Medium Risk"}
          </div>
        </div>
      </div>

      {summary ? (
        <div className="card">
          <h3>GenAI Summary</h3>
          <div style={{ lineHeight: 1.6 }}>{summary}</div>
          {reasoning.genai_note ? (
            <div className="stat-foot" style={{ marginTop: 10 }}>
              {reasoning.genai_note}
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="grid-2">
        <div className="card">
          <h3>Probable Root Cause</h3>
          <div style={{ lineHeight: 1.6 }}>
            {reasoning.probable_root_cause ||
              "Mechanical contact during wafer handling or transport. Likely caused by improper end-effector alignment or debris on handling equipment."}
          </div>
          <div className="stat-foot" style={{ marginTop: 12 }}>
            {reasoning.pattern_analysis
              ? `Pattern analysis: ${reasoning.pattern_analysis}`
              : "Affected process: Wafer Transport & Handling"}
          </div>
        </div>
        <div className="card">
          <h3>Recommended Corrective Action</h3>
          <div style={{ lineHeight: 1.6 }}>
            {reasoning.recommended_action ||
              "Inspect and recalibrate robotic handler end-effectors. Clean all contact surfaces. Verify vacuum chuck condition and alignment. Schedule preventive maintenance for transport mechanism."}
          </div>
          <div style={{ marginTop: 12, display: "flex", gap: 10 }}>
            <button className="button">Mark Acknowledged</button>
            <button className="button primary">Create Work Order</button>
          </div>
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "center" }}>
        <button className="button">Re-run Analysis</button>
      </div>
    </>
  );
}
