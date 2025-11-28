import pytest
import requests
import os
from PIL import Image

URL = "https://nwsd-api-qvcqyikbrq-ew.a.run.app/predict?save_mask=false&save_overlay=false"

@pytest.fixture
def image_file():
    filename = "test_image_valid.jpg"
    if not os.path.exists(filename):
        img = Image.new('RGB', (100, 100), color='red')
        img.save(filename)
    
    return filename

def test_send_100_requests(image_file):
    """
    Send 100 POST requests sequentially to the API with an image.
    """
    success_count = 0
    errors = []

    print(f"\nStarting to send 100 requests to {URL}...")

    for i in range(1, 101):
        try:
            with open(image_file, 'rb') as f:
                files = {'file': (image_file, f, 'image/jpeg')}

                response = requests.post(URL, files=files, timeout=30)

            if response.status_code == 200:
                success_count += 1
            else:
                errors.append(f"Req #{i}: Status {response.status_code} - {response.text}")

        except Exception as e:
            errors.append(f"Req #{i}: Exception {str(e)}")

        if i % 10 == 0:
            print(f"{i}...", end="", flush=True)

    print("\n")

    if errors:
        print("\n--- Errors encountered ---")
        for err in errors[:10]: 
            print(err)
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more errors.")

    assert success_count == 100, f"Only {success_count}/100 requests succeeded."