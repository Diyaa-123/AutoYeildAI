import { NavLink } from "react-router-dom";

const navItems = [
  { path: "/overview", label: "System Overview" },
  { path: "/ingestion", label: "Image Ingestion" },
  { path: "/defect-detection", label: "Defect Detection" },
  { path: "/explainability", label: "Explainability" },
  { path: "/root-cause", label: "Root Cause Analysis" },
  { path: "/drift-monitoring", label: "Drift Monitoring" },
  { path: "/synthetic-data", label: "Synthetic Data" },
  { path: "/auto-retraining", label: "Auto-Retraining" },
  { path: "/logs", label: "Logs & Artifacts" }
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-logo">AY</div>
        <div>
          <h1>AutoYield</h1>
          <span>AI Inspector v1.0</span>
        </div>
      </div>

      <nav className="nav">
        {navItems.map((item) => (
          <NavLink
            key={item.label}
            to={item.path}
            className={({ isActive }) =>
              `nav-item ${isActive ? "active" : ""}`
            }
          >
            <span className="nav-dot" />
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="status-pill">System Online</div>
        <div>Edge Node: LOC-01</div>
      </div>
    </aside>
  );
}
