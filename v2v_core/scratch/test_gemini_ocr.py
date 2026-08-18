import os
import cv2
import json
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not set in environment.")
    exit(1)

client = genai.Client(api_key=api_key)

class TextRegion(BaseModel):
    box_2d: list[int] = Field(description="Bounding box coordinates [ymin, xmin, ymax, xmax] normalized to 0-1000")
    text: str = Field(description="Original Turkish text transcribed from the image")
    translation: str = Field(description="English translation of the text")

class OCRResponse(BaseModel):
    regions: list[TextRegion] = Field(description="List of detected text regions")

# Generate a dummy image containing some Turkish text
print("Creating dummy image with text...")
img = np.zeros((720, 1280, 3), dtype=np.uint8)
# Add some background colors
img[200:400, 300:900] = [50, 50, 50]
cv2.putText(img, "YENI DERS: PYTHON AUTOMATION", (320, 310), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
cv2.putText(img, "ABONE OLUN", (500, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

_, encoded_img = cv2.imencode(".jpg", img)
image_bytes = encoded_img.tobytes()

print("Sending image to Gemini for OCR...")
try:
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            "Detect all Turkish text overlays, titles, and captions in the image. Exclude subtitles at the bottom."
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=OCRResponse,
            temperature=0.0
        )
    )
    print("Response text:", response.text)
    data = json.loads(response.text)
    print("Parsed JSON data successfully:")
    print(json.dumps(data, indent=2))
except Exception as e:
    print("Gemini OCR failed:", e)
