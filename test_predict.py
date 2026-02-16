"""Test that possible_diseases field is returned correctly."""

import json
import io
import http.client
import uuid
from PIL import Image
import numpy as np

# Create a test image
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
buf = io.BytesIO()
img.save(buf, format="JPEG")
buf.seek(0)
image_bytes = buf.read()

# Build multipart form data
boundary = uuid.uuid4().hex
parts = []
parts.append(f"--{boundary}")
parts.append('Content-Disposition: form-data; name="image"; filename="test.jpg"')
parts.append("Content-Type: image/jpeg")
parts.append("")

body = b""
for line in parts:
    body += (line + "\r\n").encode()
body += image_bytes + b"\r\n"
body += f"--{boundary}--\r\n".encode()

conn = http.client.HTTPConnection("localhost", 8000)
conn.request("POST", "/predict", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
resp = conn.getresponse()
data = json.loads(resp.read())
conn.close()

print("Predicted class:", data["predicted_class"])
print("Stress type:", data["stress_type"])
print("Disease name:", data["disease_name"])
print("Possible diseases:", data.get("possible_diseases"))
print("Processing time:", data["processing_time_ms"], "ms")
