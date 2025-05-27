from flask import Flask, jsonify, Response, send_file, request, abort
from flask_cors import CORS
import time, os, re, json

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/api/alerts')
def alerts():
    def event_stream():
        for i in range(10):
            # Here you could check for new data
            alerts = [
                {
                    'id': time.time(),
                    'type': 'red',
                    'text': "New alert!"
                }
            ]
            yield f"data: {json.dumps(alerts)}\n\n"
            time.sleep(5)  # send every 5 seconds

    return Response(event_stream(), mimetype="text/event-stream")

VIDEO_PATH = "./IMG_6749.MP4"

# @app.route("/api/video")
# def stream_video():
#     range_header = request.headers.get("Range", None)
#     if not range_header:
#         return send_file(VIDEO_PATH, mimetype="video/mp4")

#     try:
#         size = os.path.getsize(VIDEO_PATH)
#         byte1, byte2 = 0, None

#         m = re.search(r"bytes=(\d+)-(\d*)", range_header)
#         if m:
#             byte1 = int(m.group(1))
#             if m.group(2):
#                 byte2 = int(m.group(2))

#         length = size - byte1
#         if byte2 is not None:
#             length = byte2 - byte1 + 1

#         with open(VIDEO_PATH, "rb") as f:
#             f.seek(byte1)
#             data = f.read(length)

#         rv = Response(data, 206, mimetype="video/mp4", direct_passthrough=True)
#         rv.headers.add("Content-Range", f"bytes {byte1}-{byte1 + length - 1}/{size}")
#         rv.headers.add("Accept-Ranges", "bytes")
#         return rv
#     except Exception as e:
#         print("Streaming error:", e)
#         abort(500)

@app.route("/api/video")
def stream_video():
    def generate_video():
        try:
            with open(VIDEO_PATH, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)  # Read 1MB chunks
                    if not chunk:
                        f.seek(0)  # Restart the video when it ends
                        continue
                    yield chunk
        except Exception as e:
            print("Streaming error:", e)
            abort(500)

    return Response(generate_video(), mimetype="video/mp4")

if __name__ == '__main__':
    app.run(debug=True)
