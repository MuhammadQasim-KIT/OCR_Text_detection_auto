from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import math
import numpy as np

from src.ocr import ocr_words

@dataclass
class Pipeline:
    name: str
    preprocess_fn: Callable[[np.ndarray], np.ndarray]
    psm: int

    # scale factor used inside preprocessing resize (for box mapping back to original)
    scale: float = 1.0

    # if preprocessing includes deskew/rotation, mapping boxes back to original is not reliable
    can_map_to_original: bool = True

def score_words(words: List[Dict]) -> float:
    if not words:
        return -1e9

    confs = [w["conf"] for w in words if w["conf"] >= 0]
    if not confs:
        return -1e9

    mean_conf = float(np.mean(confs))
    n = len(words)

    texts = [w["text"] for w in words]
    one_char = sum(1 for t in texts if len(t) == 1)
    longish = sum(1 for t in texts if len(t) >= 4)

    score = 0.0
    score += mean_conf
    score += 8.0 * math.log1p(n)
    score += 2.0 * longish
    score -= 3.0 * one_char

    return score

def run_auto_ocr(
    bgr_img: np.ndarray,
    pipelines: List[Pipeline],
    *,
    min_mean_conf: float = 15.0,
    min_words: int = 3,
) -> Tuple[Pipeline, np.ndarray, List[Dict], float]:
    """
    Returns:
        (best_pipeline, best_preprocessed_img, best_words, best_score)

    If OCR is weak, best_pipeline.name will be suffixed with '__UNRELIABLE'.
    """
    best_p = None
    best_pre = None
    best_words: List[Dict] = []
    best_score = -1e18

    for p in pipelines:
        pre = p.preprocess_fn(bgr_img)
        words = ocr_words(pre, psm=p.psm)
        s = score_words(words)

        if s > best_score:
            best_p, best_pre, best_words, best_score = p, pre, words, s

    # reliability check
    if best_words:
        mean_conf = float(np.mean([w["conf"] for w in best_words]))
    else:
        mean_conf = 0.0

    if mean_conf < min_mean_conf or len(best_words) < min_words:
        # create a shallow "copy" with modified name
        best_p = Pipeline(
            name=f"{best_p.name}__UNRELIABLE",
            preprocess_fn=best_p.preprocess_fn,
            psm=best_p.psm,
            scale=best_p.scale,
            can_map_to_original=best_p.can_map_to_original
        )

    return best_p, best_pre, best_words, best_score
