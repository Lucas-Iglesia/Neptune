import logoDark from "./logo-dark.svg";
import logoLight from "./logo-light.svg";
import logo from "./neptune_logo.png";
import { useEffect, useState } from "react";
import Alert from "./Alert";

export function Welcome() {
  const [alerts, setAlerts] = useState([
    { id: 1, type: "red", text: "This is the first alert." },
    { id: 2, type: "yellow", text: "This is the second alert." },
    { id: 3, type: "green", text: "This is the third alert." },
  ])

  useEffect(() => {
    const eventSource = new EventSource("http://localhost:5000/api/alerts");

    eventSource.onmessage = (event) => {
      const newAlerts = JSON.parse(event.data);
      setAlerts((prevAlerts) => [...newAlerts, ...prevAlerts]);
    };

    eventSource.onerror = (error) => {
      console.error("EventSource failed:", error);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, []);

  function handleDelete(id: number) {
    setAlerts((prevAlerts) => prevAlerts.filter((alert) => alert.id !== id));
  }

  return (
    <main className="flex flex-col pt-4 pb-4 gap-8 px-8">
      <header className="flex flex-col items-center gap-9">
        <div className="w-[400px] max-w-full p-4 flex items-center justify-center gap-4">
          <img
            src={logo}
            alt="Logo Netune"
            className="block w-full"
          />
          <span className="text-6xl font-bold text-white dark:text-white">Neptune</span>
        </div>
      </header>
      <div className="flex flex-row w-full gap-8 items-stretch">
        <section className="flex-grow">
          <div className="relative pb-[56.25%] h-0 overflow-hidden rounded-lg shadow-lg">
            <video
              className="w-full rounded-lg shadow-lg"
              autoPlay
              loop
              muted
              preload="metadata"
            >
              <source src="http://localhost:5000/api/video" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>
        </section>

        <aside className="w-[400px] relative">
          <div className="absolute inset-0 flex flex-col gap-4 bg-white/5 p-4 rounded-lg overflow-y-auto">
            <button
              onClick={() => setAlerts([])}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >
              Clear Alerts
            </button>
            {alerts.map((alert) => (
              <Alert
                key={alert.id}
                id={alert.id}
                type={alert.type as "red" | "yellow" | "green"}
                text={alert.text}
                onDelete={handleDelete}
              />
            ))}
          </div>
        </aside>
      </div>
    </main>
  );
}
