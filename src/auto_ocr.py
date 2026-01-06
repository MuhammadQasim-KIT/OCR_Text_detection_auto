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
) -> Tuple[str, np.ndarray, List[Dict], float]:

    best_name = ""
    best_pre = None
    best_words: List[Dict] = []
    best_score = -1e18

    for p in pipelines:
        pre = p.preprocess_fn(bgr_img)
        words = ocr_words(pre, psm=p.psm)
        s = score_words(words)
        if s > best_score:
            best_name, best_pre, best_words, best_score = p.name, pre, words, s

    if best_words:
        mean_conf = float(np.mean([w["conf"] for w in best_words]))
    else:
        mean_conf = 0.0

    if mean_conf < min_mean_conf or len(best_words) < min_words:
        best_name = f"{best_name}__UNRELIABLE"

    return best_name, best_pre, best_words, best_score
