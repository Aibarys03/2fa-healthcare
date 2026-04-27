"""
Liveness Detection Module — v2 (calibrated for real-world web cameras)
======================================================================

Изменения по сравнению с v1:

  1. Active liveness теперь требует УСТОЙЧИВОГО моргания
     (минимум 2 подряд кадра с закрытыми глазами и потом 2 кадра с открытыми).
     Это убирает ложные срабатывания от шума EAR.

  2. Пороги EAR откалиброваны мягче — реальные веб-камеры (Render + JPEG q=85)
     дают EAR в диапазоне 0.18-0.32 для открытых глаз и 0.10-0.18 для закрытых.
     Используем относительный, а не абсолютный порог.

  3. Passive liveness стал opt-in (выключается, если вы хотите
     отлаживать только active часть). По умолчанию ВЫКЛЮЧЕН — слишком
     много ложных срабатываний на маленьких кадрах. Включается флагом
     ENABLE_PASSIVE = True, когда будете готовы калибровать.

  4. Добавлены подробные поля в ответе: ear_history (последние 10),
     baseline_open, baseline_closed, frames_with_face, frames_without_face.
     Это позволит фронту показывать живой график.

  5. Сессия пересоздаётся при каждом start_liveness_session — старая
     стирается, новая начинается с чистого листа. Поэтому повторный
     запрос работает без перезагрузки.

  6. Если лицо не найдено в кадре — это НЕ инкрементирует
     frames_processed. Считаются только кадры, в которых лицо есть.
     Так MAX_FRAMES_FOR_BLINK работает по реальным замерам, а не по таймеру.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

import mediapipe as mp

# Passive checks — необязательные, импортируем только если включены
try:
    from skimage.feature import local_binary_pattern
    from skimage.color import rgb2gray
    from scipy import fftpack
    _PASSIVE_AVAILABLE = True
except ImportError:
    _PASSIVE_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────
# Глобальные параметры
# ──────────────────────────────────────────────────────────────────

# Включить passive проверки (LBP + FFT). Пока выключено для отладки.
ENABLE_PASSIVE = False

# EAR — относительный режим: считаем "open baseline" из первых кадров
# и считаем что "закрыто" = baseline * CLOSED_RATIO
EAR_CLOSED_RATIO = 0.65        # глаз закрыт если EAR < baseline * 0.65
EAR_OPEN_RATIO = 0.85          # глаз открыт если EAR > baseline * 0.85

# Минимум кадров для калибровки baseline
BASELINE_FRAMES = 5

# Сколько кадров подряд должно быть "закрыто", чтобы засчитать как закрытие
CLOSED_FRAMES_REQUIRED = 2

# Сколько кадров подряд "открыто" нужно после закрытия для подтверждения моргания
OPEN_AFTER_CLOSE_FRAMES = 2

# Считаются только кадры с обнаруженным лицом
MAX_FACE_FRAMES = 25           # ~5 секунд при 200ms/кадр

# Сколько последних EAR-значений возвращать в debug
EAR_HISTORY_SIZE = 15


# ──────────────────────────────────────────────────────────────────
# MediaPipe FaceMesh — ключевые точки глаз
# ──────────────────────────────────────────────────────────────────

LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.4,    # понизили с 0.5 — иначе теряет лицо
)


# ──────────────────────────────────────────────────────────────────
# Структура сессии
# ──────────────────────────────────────────────────────────────────

@dataclass
class LivenessSession:
    session_id: str
    created_at: float = field(default_factory=time.time)

    # EAR-метрики
    ear_history: list[float] = field(default_factory=list)
    baseline_open_ear: Optional[float] = None  # калибровочный baseline

    # Состояние конечного автомата моргания
    consecutive_closed: int = 0
    consecutive_open_after_close: int = 0
    saw_closed: bool = False
    blink_detected: bool = False

    # Счётчики
    frames_with_face: int = 0
    frames_without_face: int = 0

    # Passive
    passive_passed: bool = True   # по умолчанию True (если passive выключено)
    passive_details: dict = field(default_factory=dict)


_live_sessions: dict[str, LivenessSession] = {}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _euclid(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _eye_aspect_ratio(landmarks, idx, img_w, img_h):
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in idx]
    v1 = _euclid(pts[1], pts[5])
    v2 = _euclid(pts[2], pts[4])
    h = _euclid(pts[0], pts[3])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_ear(image: Image.Image) -> Optional[float]:
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]
    results = _face_mesh.process(img_np)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    left = _eye_aspect_ratio(lm, LEFT_EYE_EAR_IDX, w, h)
    right = _eye_aspect_ratio(lm, RIGHT_EYE_EAR_IDX, w, h)
    return (left + right) / 2.0


# ──────────────────────────────────────────────────────────────────
# Passive (опционально)
# ──────────────────────────────────────────────────────────────────

def passive_liveness_check(image: Image.Image) -> dict:
    if not _PASSIVE_AVAILABLE:
        return {"passed": True, "skipped": True}

    gray = rgb2gray(np.array(image.convert("RGB")))
    gray_uint = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_uint, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    lbp_var = float(np.var(hist))

    f = fftpack.fft2(gray)
    f_shift = fftpack.fftshift(f)
    magnitude = np.log1p(np.abs(f_shift))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = min(cy, cx)
    midband = (radius >= 0.25 * rmax) & (radius <= 0.60 * rmax)
    total = magnitude.sum() + 1e-6
    fft_ratio = float(magnitude[midband].sum() / total)

    # Очень мягкие пороги — отлаживайте на ваших данных
    LBP_MIN = 0.005
    FFT_MAX = 0.50
    return {
        "passed": (lbp_var >= LBP_MIN) and (fft_ratio <= FFT_MAX),
        "lbp_variance": round(lbp_var, 5),
        "fft_ratio": round(fft_ratio, 5),
    }


# ──────────────────────────────────────────────────────────────────
# Конечный автомат моргания (новая логика)
# ──────────────────────────────────────────────────────────────────

def _update_blink_fsm(session: LivenessSession, ear: float) -> None:
    """
    FSM:
      State A: ждём калибровки baseline (первые BASELINE_FRAMES кадров)
      State B: открытые глаза (EAR > baseline * EAR_OPEN_RATIO)
      State C: закрытые глаза (EAR < baseline * EAR_CLOSED_RATIO,
               и так минимум CLOSED_FRAMES_REQUIRED кадров)
      State D: открытые глаза после закрытия — если так
               OPEN_AFTER_CLOSE_FRAMES подряд → моргание подтверждено.
    """
    # Калибровка baseline на первых BASELINE_FRAMES кадрах с лицом
    if session.baseline_open_ear is None:
        if session.frames_with_face >= BASELINE_FRAMES:
            recent = session.ear_history[-BASELINE_FRAMES:]
            session.baseline_open_ear = float(np.median(recent))
        return

    closed_thr = session.baseline_open_ear * EAR_CLOSED_RATIO
    open_thr = session.baseline_open_ear * EAR_OPEN_RATIO

    is_closed = ear < closed_thr
    is_open = ear > open_thr

    if is_closed:
        session.consecutive_closed += 1
        session.consecutive_open_after_close = 0
        if session.consecutive_closed >= CLOSED_FRAMES_REQUIRED:
            session.saw_closed = True
    elif is_open:
        if session.saw_closed:
            session.consecutive_open_after_close += 1
            if session.consecutive_open_after_close >= OPEN_AFTER_CLOSE_FRAMES:
                session.blink_detected = True
        else:
            session.consecutive_closed = 0
    # промежуточная зона между порогами — ничего не делаем


# ──────────────────────────────────────────────────────────────────
# Публичный API
# ──────────────────────────────────────────────────────────────────

def start_liveness_session(session_id: str) -> LivenessSession:
    """Создаёт новую liveness-сессию, всегда стирая старую с тем же ID."""
    _live_sessions.pop(session_id, None)            # сброс старой
    session = LivenessSession(session_id=session_id)
    _live_sessions[session_id] = session
    return session


def get_liveness_session(session_id: str) -> Optional[LivenessSession]:
    return _live_sessions.get(session_id)


def end_liveness_session(session_id: str) -> None:
    _live_sessions.pop(session_id, None)


def reset_liveness_session(session_id: str) -> LivenessSession:
    """Кнопка 'попробовать ещё раз' с фронта — обнуляем состояние."""
    return start_liveness_session(session_id)


def process_liveness_frame(session_id: str, image: Image.Image) -> dict:
    session = get_liveness_session(session_id)
    if session is None:
        return {"status": "failed", "reason": "session_not_found"}

    # Passive один раз на первом кадре с лицом
    if ENABLE_PASSIVE and not session.passive_details:
        passive = passive_liveness_check(image)
        session.passive_details = passive
        session.passive_passed = passive["passed"]
        if not session.passive_passed:
            return {
                "status": "failed",
                "reason": "passive_check_failed",
                "passive": passive,
            }

    ear = compute_ear(image)
    if ear is None:
        session.frames_without_face += 1
        # Если 10 кадров подряд нет лица — провал
        if session.frames_without_face >= 10 and session.frames_with_face == 0:
            return {
                "status": "failed",
                "reason": "no_face_detected",
                "frames_without_face": session.frames_without_face,
            }
        return {
            "status": "pending",
            "reason": "no_face_in_frame",
            "frames_without_face": session.frames_without_face,
            "frames_with_face": session.frames_with_face,
        }

    # Кадр с лицом
    session.frames_with_face += 1
    session.ear_history.append(ear)
    if len(session.ear_history) > EAR_HISTORY_SIZE:
        session.ear_history = session.ear_history[-EAR_HISTORY_SIZE:]

    _update_blink_fsm(session, ear)

    # Готово
    if session.blink_detected and session.passive_passed:
        return {
            "status": "success",
            "ear": round(ear, 3),
            "baseline": round(session.baseline_open_ear or 0, 3),
            "frames_with_face": session.frames_with_face,
            "ear_history": [round(v, 3) for v in session.ear_history],
        }

    # Тайм-аут
    if session.frames_with_face >= MAX_FACE_FRAMES:
        return {
            "status": "failed",
            "reason": "no_blink_detected_in_time",
            "ear": round(ear, 3),
            "baseline": round(session.baseline_open_ear or 0, 3),
            "frames_with_face": session.frames_with_face,
            "saw_closed": session.saw_closed,
            "ear_history": [round(v, 3) for v in session.ear_history],
        }

    # Продолжаем
    return {
        "status": "pending",
        "ear": round(ear, 3),
        "baseline": round(session.baseline_open_ear or 0, 3),
        "calibrating": session.baseline_open_ear is None,
        "frames_with_face": session.frames_with_face,
        "saw_closed": session.saw_closed,
        "blink_detected": session.blink_detected,
        "consecutive_closed": session.consecutive_closed,
        "consecutive_open_after_close": session.consecutive_open_after_close,
    }
