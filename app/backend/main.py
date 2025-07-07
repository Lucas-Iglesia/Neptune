from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2, json, time

from homography.processor import HomographyProcessor

VIDEO_PATH = "assets/input.mov"
JPEG_QUALITY = 50    

app = Flask(__name__)
CORS(app)

hp = HomographyProcessor(VIDEO_PATH)

# ---------- MJPEG stream : /api/video ----------
def mjpeg_stream():
    for frame in hp.frames():

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
            alerts = [{
                "id": time.time(),
                "type": "red",
                "text": "New alert!"
            }]
            yield f"data: {json.dumps(alerts)}\n\n"
            time.sleep(5)

    return Response(event_stream(),
                    mimetype="text/event-stream")

@app.route("/")
def index():
    return "Backend up 👍"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
