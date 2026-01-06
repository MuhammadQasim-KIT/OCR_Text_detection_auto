from typing import Dict, List
import cv2
import numpy as np

def draw_boxes(bgr_img: np.ndarray, words: List[Dict]) -> np.ndarray:
    for w in words:
        x, y, bw, bh = w["x"], w["y"], w["w"], w["h"]

        cv2.rectangle(bgr_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        label = f'{w["text"]} ({w["conf"]:.0f})'
        cv2.putText(
            bgr_img,
            label,
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
    return bgr_img
