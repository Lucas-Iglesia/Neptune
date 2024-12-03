import os
import subprocess
import threading
from datetime import datetime
import time
import signal

def record_youtube_live(url, output_folder):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(output_folder, f"live_{timestamp}.mp4")

        command = [
            "yt-dlp",
            "--no-check-certificate",
            "-o", output_file,
            url
        ]

        print(f"Starting recording for {url}...")

        process = subprocess.Popen(command)

        print("Press 'q' to stop recording.")

        def listen_for_exit():
            while True:
                user_input = input()
                if user_input.lower() == 'q':
                    print("\nStop requested, stopping recording...")
                    process.send_signal(signal.SIGINT)
                    process.wait()
                    print(f"Recording finished: {output_file}")
                    exit(0)

        input_thread = threading.Thread(target=listen_for_exit)
        input_thread.daemon = True
        input_thread.start()

        while process.poll() is None:
            time.sleep(1)

        print(f"Process finished normally: {output_file}")

    except subprocess.CalledProcessError as e:
        print("Error during recording:", e)
    except Exception as e:
        print("General error:", e)

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=mvVoilECpoY&embeds_referring_euri=https%3A%2F%2Fwww.skylinewebcams.com%2F&embeds_referring_origin=https%3A%2F%2Fwww.skylinewebcams.com&source_ve_path=MjM4NTE"
    output_directory = "./recordings"

    os.makedirs(output_directory, exist_ok=True)

    record_youtube_live(youtube_url, output_directory)