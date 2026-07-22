"""
Project 4: Image or Text Recognition (Basic)  -- Path 1: OCR
DecodeLabs Industrial Training Kit, Batch 2026
=============================================================



"""

import sys
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# ----------------------------------------------------------------------
# CONFIG -- Page 16: "In Project 4, 80% is the absolute minimum standard"
# ----------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 80  # percent
INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "sample_input.png"
OUTPUT_PATH = "output_annotated.png"


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Step 3 (Page 11): Deskewing.
    Calculates the rotation angle of the text mass and rotates the
    image back to a horizontal baseline.
    """
    # Invert + threshold so text pixels are white on a black background,
    # which is what cv2.minAreaRect expects to find contours of.
    inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return gray  # nothing detected, skip deskew

    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angles in [-90, 0); normalize to a small
    # signed rotation rather than a full 90-degree flip.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    The Logic Skeleton (Page 11) + Adaptive Thresholding (Page 12).

    Step 1: Grayscale conversion  -> collapse 3D RGB matrix to 1D intensity
    Step 2: Gaussian blur         -> smooth noise/artifacts
    Step 3: Deskew                -> straighten tilted text line
    Step 4: Otsu thresholding     -> force every pixel to pure black/white
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                 # Step 1
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)                    # Step 2
    straightened = deskew(blurred)                                 # Step 3

    # Step 4: Otsu's method automatically picks the cutoff intensity
    # (Page 12 illustrates this with a manual cutoff of 88; Otsu computes
    # the equivalent optimal cutoff automatically instead of hardcoding it).
    _, binary = cv2.threshold(
        straightened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def run_ocr(binary_image: np.ndarray):
    """
    Path 1 (Page 9-10): pytesseract with --psm 6, tuned for a single
    uniform block of text (appropriate for a document/invoice-style input).
    Returns pytesseract's word-level dataframe-like dict, which includes
    a confidence score per detected word (Page 15: Softmax & Confidence).
    """
    config = "--psm 6"
    data = pytesseract.image_to_data(
        binary_image, config=config, output_type=Output.DICT
    )
    return data


def filter_and_annotate(original_bgr: np.ndarray, binary_image: np.ndarray, data: dict):
    """
    The 80% Threshold / Confidence Filter (Page 16):
        if confidence >= 0.80: draw_box_and_label()
        else: drop_detection()

    Draws bounding boxes + labels only for words that clear the gate,
    and returns the clean recognized text string alongside the
    annotated image.
    """
    annotated = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    accepted_words = []
    n = len(data["text"])

    for i in range(n):
        word = data["text"][i].strip()
        conf = int(float(data["conf"][i])) if data["conf"][i] not in ("-1", "") else -1

        if not word or conf < 0:
            continue  # skip empty detections

        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

        if conf >= CONFIDENCE_THRESHOLD:
            accepted_words.append(word)
            color = (0, 200, 0)  # green = accepted
            label = f"{word} ({conf}%)"
        else:
            color = (0, 0, 255)  # red = dropped (below gate)
            label = f"dropped ({conf}%)"

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            annotated, label, (x, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )

    return " ".join(accepted_words), annotated


def main():
    print("=" * 60)
    print("PROJECT 4: IMAGE OR TEXT RECOGNITION (BASIC) -- Path 1: OCR")
    print("=" * 60)

    # 1. LIBRARY INTEGRATION: load image, hand off to pytesseract pipeline
    original = cv2.imread(INPUT_PATH)
    if original is None:
        raise FileNotFoundError(f"Could not read input image: {INPUT_PATH}")
    print(f"[1/4] Loaded input image: {INPUT_PATH}  shape={original.shape}")

    # 2. PRE-PROCESSING INTEGRITY
    binary_image = preprocess(original)
    print("[2/4] Pre-processing complete: grayscale -> blur -> deskew -> Otsu threshold")

    # 3. ACCURACY BENCHMARKING
    data = run_ocr(binary_image)
    recognized_text, annotated = filter_and_annotate(original, binary_image, data)
    print(f"[3/4] OCR complete. Confidence gate set at {CONFIDENCE_THRESHOLD}%")

    # 4. VISUAL CONFIRMATION
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"[4/4] Annotated output saved -> {OUTPUT_PATH}")

    print("\n--- RECOGNIZED TEXT (confidence >= {}%) ---".format(CONFIDENCE_THRESHOLD))
    print(recognized_text if recognized_text else "(no text cleared the confidence gate)")
    print("-" * 60)


if __name__ == "__main__":
    main()
