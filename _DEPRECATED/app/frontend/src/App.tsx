import { useEffect, useState } from "react";
import Alert from "./components/Alert";
import VideoPlayer from "./components/VideoPlayer";
import Logo from "./components/Logo";

export default function App() {
  const [alerts, setAlerts] = useState<
    { id: number; type: "red" | "yellow" | "green"; text: string; playAudio?: boolean }[]
  >([]);

  useEffect(() => {
    const es = new EventSource("http://localhost:5000/api/alerts");

    es.onmessage = (ev) => {
      const incoming = JSON.parse(ev.data);
      setAlerts((prev) => [...incoming, ...prev]);
    };

    es.onerror = (e) => {
      console.error("SSE error :", e);
      es.close();
    };
    return () => es.close();
  }, []);

  const handleDelete = (id: number) =>
    setAlerts((prev) => prev.filter((a) => a.id !== id));

  return (
    <div className="app">
      <header className="header">
        <Logo />
      </header>

      <div className="main-row">
        {/* Vidéo */}
        <section className="video-container">
          <VideoPlayer />
        </section>

        {/* Alertes */}
        <aside className="alerts-panel">
          <button className="clear-btn" onClick={() => setAlerts([])}>
            Clear Alerts
          </button>
          <div className="alerts-scroll">
            {alerts.map((a) => (
              <Alert key={a.id} {...a} onDelete={handleDelete} />
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}
