"""
generate_sample.py
-------------------
Creates a realistic 'messy scanned document' sample image for Project 4.
Simulates the exact problems the slide deck describes on Page 11:
  - uneven lighting / shadow gradient
  - chromatic (color) noise
  - a slightly tilted text line

This gives the pre-processing pipeline (grayscale -> blur -> threshold ->
deskew) something real to do, instead of feeding pytesseract a perfect image.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 900, 300

# 1. Base canvas with an uneven lighting gradient (simulates shadow/glare)
gradient = np.tile(np.linspace(200, 255, W), (H, 1)).astype(np.uint8)
base = np.stack([gradient, gradient, gradient], axis=-1)

img = Image.fromarray(base, mode="RGB")
draw = ImageDraw.Draw(img)

# 2. Draw the text ("scanned document" content)
try:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48
    )
except OSError:
    font = ImageFont.load_default()

text = "DECODELABS INVOICE 2026"
draw.text((60, 110), text, fill=(20, 20, 20), font=font)

# 3. Add chromatic (color) noise
arr = np.array(img).astype(np.int16)
noise = np.random.randint(-25, 25, arr.shape, dtype=np.int16)
arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
img = Image.fromarray(arr, mode="RGB")

# 4. Slight rotation to simulate a tilted scan (Page 11: "Deskewing")
img = img.rotate(4, expand=True, fillcolor=(230, 230, 230))

img.save("sample_input.png")
print("Saved sample_input.png", img.size)
