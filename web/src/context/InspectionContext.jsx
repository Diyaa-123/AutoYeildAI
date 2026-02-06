import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { analyzeImage, getHistory, getMetrics } from "../api/client.js";

const InspectionContext = createContext(null);

export function InspectionProvider({ children }) {
  const [inspection, setInspection] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState(null);

  const runAnalysis = async (file, options) => {
    setLoading(true);
    setError("");
    try {
      const result = await analyzeImage(file, options);
      setInspection(result);
      const [historyData, metricsData] = await Promise.all([
        getHistory(),
        getMetrics()
      ]);
      setHistory(historyData);
      setMetrics(metricsData);
      return result;
    } catch (err) {
      setError(err.message || "Analysis failed");
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const refreshDashboard = async () => {
    try {
      const [historyData, metricsData] = await Promise.all([
        getHistory(),
        getMetrics()
      ]);
      setHistory(historyData);
      setMetrics(metricsData);
    } catch (err) {
      setError(err.message || "Failed to load dashboard data");
    }
  };

  useEffect(() => {
    refreshDashboard();
  }, []);

  const value = useMemo(
    () => ({
      inspection,
      setInspection,
      runAnalysis,
      loading,
      error,
      history,
      metrics,
      refreshDashboard
    }),
    [inspection, loading, error, history, metrics]
  );

  return (
    <InspectionContext.Provider value={value}>
      {children}
    </InspectionContext.Provider>
  );
}

export function useInspection() {
  const context = useContext(InspectionContext);
  if (!context) {
    throw new Error("useInspection must be used within InspectionProvider");
  }
  return context;
}
