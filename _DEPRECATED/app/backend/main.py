from flask import Flask, Response, jsonify, send_file
from flask_cors import CORS
import cv2, json, time
import threading
import queue

from homography.processor import HomographyProcessor

VIDEO_PATH = "assets/input.MOV"
JPEG_QUALITY = 80

app = Flask(__name__)
CORS(app)

hp = HomographyProcessor(VIDEO_PATH)

# Global alert queue for SSE
alert_queue = queue.Queue()

# ---------- MJPEG stream : /api/video ----------
def mjpeg_stream():
    for frame in hp.frames():
        # Check for alerts during frame processing
        alerts = hp.get_alerts()
        for alert in alerts:
            alert_queue.put(alert)
            print(f"🚨 Alert added to queue: {alert}")

        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not ok:
            continue
        jpg = buf.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

@app.route("/api/video")
def video_feed():
    return Response(mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ---------- Server-Sent Events : /api/alerts ----------
@app.route("/api/alerts")
def alerts():
    def event_stream():
        while True:
            try:
                # Get alert from queue with timeout
                alert = alert_queue.get(timeout=1)
                
                # Format alert for frontend
                frontend_alert = {
                    "id": int(alert["timestamp"] * 1000),  # Use timestamp as ID
                    "type": "red",  # All danger alerts are red
                    "text": alert["message"],
                    "playAudio": True,  # Trigger audio playback
                    "duration": alert["duration"],
                    "track_id": alert["track_id"]
                }
                
                yield f"data: {json.dumps([frontend_alert])}\n\n"
                
            except queue.Empty:
                # Send heartbeat to keep connection alive
                yield "data: []\n\n"

    return Response(event_stream(),
                    mimetype="text/event-stream")

# ---------- Static files : /api/alert-audio ----------
@app.route("/api/alert-audio")
def alert_audio():
    try:
        return send_file("assets/alert_audio.mp3", mimetype="audio/mpeg")
    except Exception as e:
        return f"Error serving audio: {e}", 404

@app.route("/")
def index():
    return "Backend up 👍"

# ---------- Debug routes ----------
@app.route("/api/debug/tracks")
def debug_tracks():
    """Debug endpoint to see current tracks"""
    try:
        tracks_info = []
        for tid, track in hp.tracker.tracks.items():
            tracks_info.append({
                "id": tid,
                "status": track["status"],
                "frames_underwater": track["frames_underwater"],
                "underwater_duration": track.get("underwater_duration", 0),
                "danger_alert_sent": track["danger_alert_sent"],
                "dangerosity_score": track["dangerosity_score"]
            })
        return jsonify({
            "tracks": tracks_info,
            "queue_size": alert_queue.qsize()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/debug/test-alert")
def test_alert():
    """Test endpoint to generate a fake alert"""
    test_alert = {
        "track_id": 999,
        "type": "danger", 
        "message": "Test alert - Personne 999 en danger",
        "duration": 10.5,
        "timestamp": time.time()
    }
    alert_queue.put(test_alert)
    return jsonify({"message": "Test alert sent", "alert": test_alert})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
