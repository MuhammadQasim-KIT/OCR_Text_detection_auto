from typing import Dict, List
import numpy as np
import pytesseract
from pytesseract import Output

def ocr_words(preprocessed_img: np.ndarray, psm: int = 6) -> List[Dict]:
    """
    Run word-level OCR on a preprocessed (binary/grayscale) image.

    Args:
        preprocessed_img: single-channel image (uint8)
        psm: Tesseract Page Segmentation Mode

    Returns:
        A list of dicts with:
        - text (str)
        - conf (float)
        - x, y, w, h (int)
    """
    config = fr"--oem 3 --psm {psm}"

    data = pytesseract.image_to_data(
        preprocessed_img,
        output_type=Output.DICT,
        config=config
    )

    results: List[Dict] = []
    n = len(data.get("text", []))

    for i in range(n):
        text = (data["text"][i] or "").strip()

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        # Filter empty, tiny, low-confidence
        if not text:
            continue
        if w <= 5 or h <= 5:
            continue
        if conf < 20:
            continue

        results.append({
            "text": text,
            "conf": conf,
            "x": x, "y": y, "w": w, "h": h
        })

    return results
