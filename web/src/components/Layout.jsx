import Sidebar from "./Sidebar.jsx";
import Topbar from "./Topbar.jsx";

export default function Layout({ children }) {
  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main mini-grid">
        <Topbar />
        <div className="content">{children}</div>
      </main>
    </div>
  );
}
