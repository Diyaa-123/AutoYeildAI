import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import SystemOverview from "./pages/SystemOverview.jsx";
import ImageIngestion from "./pages/ImageIngestion.jsx";
import DefectDetection from "./pages/DefectDetection.jsx";
import Explainability from "./pages/Explainability.jsx";
import RootCause from "./pages/RootCause.jsx";
import DriftMonitoring from "./pages/DriftMonitoring.jsx";
import SyntheticData from "./pages/SyntheticData.jsx";
import AutoRetraining from "./pages/AutoRetraining.jsx";
import LogsArtifacts from "./pages/LogsArtifacts.jsx";
import { InspectionProvider } from "./context/InspectionContext.jsx";

export default function App() {
  return (
    <InspectionProvider>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<SystemOverview />} />
          <Route path="/ingestion" element={<ImageIngestion />} />
          <Route path="/defect-detection" element={<DefectDetection />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/root-cause" element={<RootCause />} />
          <Route path="/drift-monitoring" element={<DriftMonitoring />} />
          <Route path="/synthetic-data" element={<SyntheticData />} />
          <Route path="/auto-retraining" element={<AutoRetraining />} />
          <Route path="/logs" element={<LogsArtifacts />} />
          <Route path="*" element={<Navigate to="/overview" replace />} />
        </Routes>
      </Layout>
    </InspectionProvider>
    
  );
}
