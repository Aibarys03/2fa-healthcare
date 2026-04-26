"""
Liveness Detection Module
==========================
Гибридная защита от презентационных атак (ISO/IEC 30107-3):

  1. Active (blink-based)
     - Серия из N кадров, на которых пользователь должен моргнуть.
     - Eye Aspect Ratio (EAR) рассчитывается из 6 точек глаза;
       моргание = резкое падение EAR с последующим восстановлением.

  2. Passive (texture / spoof artefacts)
     - Анализ Local Binary Patterns (LBP) — у настоящих лиц распределение
       LBP-кодов отличается от распечатанных/экранных копий.
     - Анализ Frequency Domain (FFT) — у экранных копий выражены
       Moiré-паттерны и резкие пики на средних частотах.

Оба сигнала независимы: атакующему нужно обойти оба одновременно.

Зависимости (добавить в requirements.txt):
    mediapipe==0.10.14
    scipy==1.13.1
    scikit-image==0.24.0

Установка:
    pip install mediapipe scipy scikit-image
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

# MediaPipe для FaceMesh (468 точек лица, в т.ч. вокруг глаз)
import mediapipe as mp

# scikit-image для LBP
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

# scipy для FFT
from scipy import fftpack


# ──────────────────────────────────────────────────────────────────
# Параметры
# ──────────────────────────────────────────────────────────────────

# EAR-индексы согласно официальной разметке MediaPipe FaceMesh
# (https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh.py)
LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]   # p1..p6 левого глаза
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380] # p1..p6 правого глаза

EAR_BLINK_THRESHOLD = 0.21      # ниже этого значения = глаз закрыт
EAR_OPEN_THRESHOLD = 0.27       # выше этого = глаз открыт
MIN_FRAMES_FOR_BLINK = 5        # минимум кадров для регистрации моргания
MAX_FRAMES_FOR_BLINK = 30       # максимум кадров до тайм-аута

LBP_RADIUS = 1
LBP_POINTS = 8
LBP_HISTOGRAM_BINS = LBP_POINTS + 2

# Эмпирические пороги, откалиброванные на CASIA-FASD-подобных данных
LBP_REAL_VARIANCE_MIN = 0.012   # настоящие лица имеют большую вариативность
FFT_SPOOF_RATIO_MAX = 0.35      # доля энергии в средних частотах (Moiré)


# ──────────────────────────────────────────────────────────────────
# Структуры
# ──────────────────────────────────────────────────────────────────

@dataclass
class LivenessSession:
    """
    Состояние одной активной сессии проверки живости.
    Сохраняется по session_id (тот же, что выдаётся после face verification).
    """
    session_id: str
    ear_history: list[float] = field(default_factory=list)
    blink_detected: bool = False
    frames_processed: int = 0
    passive_passed: bool = False


# Глобальное хранилище живости-сессий (в production — Redis/Supabase)
_live_sessions: dict[str, LivenessSession] = {}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _euclid(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _eye_aspect_ratio(landmarks, idx: list[int], img_w: int, img_h: int) -> float:
    """
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    Soukupová & Čech, 2016 — стандартная формула.
    """
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in idx]
    vertical_1 = _euclid(pts[1], pts[5])
    vertical_2 = _euclid(pts[2], pts[4])
    horizontal = _euclid(pts[0], pts[3])
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# Один глобальный объект FaceMesh — переиспользуем между запросами
_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,         # статические кадры (а не видеопоток)
    max_num_faces=1,
    refine_landmarks=True,          # точные точки вокруг глаз и рта
    min_detection_confidence=0.5,
)


# ──────────────────────────────────────────────────────────────────
# Active liveness — детектор моргания
# ──────────────────────────────────────────────────────────────────

def compute_ear(image: Image.Image) -> Optional[float]:
    """
    Возвращает усреднённый EAR (левый+правый) или None, если лицо не найдено.
    """
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]

    results = _face_mesh.process(img_np)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE_EAR_IDX, w, h)
    right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE_EAR_IDX, w, h)
    return (left_ear + right_ear) / 2.0


def update_blink_state(session: LivenessSession, ear: float) -> bool:
    """
    Добавляет EAR в историю и проверяет, было ли моргание.
    Моргание = последовательность: открыт → закрыт → открыт.
    """
    session.ear_history.append(ear)
    session.frames_processed += 1

    history = session.ear_history
    if len(history) < 3:
        return False

    # Сканируем историю на паттерн open → close → open
    has_open_before = False
    has_closed = False
    for value in history:
        if not has_open_before and value > EAR_OPEN_THRESHOLD:
            has_open_before = True
            continue
        if has_open_before and not has_closed and value < EAR_BLINK_THRESHOLD:
            has_closed = True
            continue
        if has_open_before and has_closed and value > EAR_OPEN_THRESHOLD:
            session.blink_detected = True
            return True
    return False


# ──────────────────────────────────────────────────────────────────
# Passive liveness — текстурный + частотный анализ
# ──────────────────────────────────────────────────────────────────

def _lbp_variance(image: Image.Image) -> float:
    """
    Вариация распределения LBP — у напечатанных/экранных копий
    распределение более «гладкое» (низкая вариация).
    """
    gray = rgb2gray(np.array(image.convert("RGB")))
    gray = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_HISTOGRAM_BINS,
                           range=(0, LBP_HISTOGRAM_BINS), density=True)
    return float(np.var(hist))


def _fft_midband_ratio(image: Image.Image) -> float:
    """
    Доля энергии спектра в среднем частотном кольце.
    У экранных Moiré-копий эта доля выше, чем у живых лиц.
    Чем выше — тем подозрительнее.
    """
    gray = rgb2gray(np.array(image.convert("RGB")))
    f = fftpack.fft2(gray)
    f_shift = fftpack.fftshift(f)
    magnitude = np.log1p(np.abs(f_shift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    # Среднее кольцо: 25–60% от max радиуса
    rmax = min(cy, cx)
    midband = (radius >= 0.25 * rmax) & (radius <= 0.60 * rmax)
    total = magnitude.sum()
    if total < 1e-6:
        return 0.0
    return float(magnitude[midband].sum() / total)


def passive_liveness_check(image: Image.Image) -> dict:
    """
    Возвращает {'passed': bool, 'lbp_variance': float, 'fft_ratio': float}.
    """
    lbp_var = _lbp_variance(image)
    fft_ratio = _fft_midband_ratio(image)

    lbp_ok = lbp_var >= LBP_REAL_VARIANCE_MIN
    fft_ok = fft_ratio <= FFT_SPOOF_RATIO_MAX

    return {
        "passed": bool(lbp_ok and fft_ok),
        "lbp_variance": round(lbp_var, 5),
        "lbp_passed": bool(lbp_ok),
        "fft_ratio": round(fft_ratio, 5),
        "fft_passed": bool(fft_ok),
    }


# ──────────────────────────────────────────────────────────────────
# Публичный API: используется из app.py
# ──────────────────────────────────────────────────────────────────

def start_liveness_session(session_id: str) -> LivenessSession:
    """Создаёт новую liveness-сессию (вызывается после face verification)."""
    session = LivenessSession(session_id=session_id)
    _live_sessions[session_id] = session
    return session


def get_liveness_session(session_id: str) -> Optional[LivenessSession]:
    return _live_sessions.get(session_id)


def end_liveness_session(session_id: str) -> None:
    _live_sessions.pop(session_id, None)


def process_liveness_frame(session_id: str, image: Image.Image) -> dict:
    """
    Главная функция: обрабатывает один кадр и возвращает текущее состояние
    проверки живости. Вызывается с фронтенда раз в ~200 мс.

    Состояния возврата:
      - 'pending'   — продолжаем съёмку, ждём моргание
      - 'success'   — обе проверки (active + passive) прошли
      - 'failed'    — превышено число кадров либо лицо не найдено
    """
    session = get_liveness_session(session_id)
    if session is None:
        return {"status": "failed", "reason": "session_not_found"}

    # Passive — выполняем один раз на первом кадре (texture не меняется
    # принципиально, и это снижает нагрузку)
    if not session.passive_passed and session.frames_processed == 0:
        passive = passive_liveness_check(image)
        session.passive_passed = passive["passed"]
        if not passive["passed"]:
            return {
                "status": "failed",
                "reason": "passive_check_failed",
                "details": passive,
            }

    # Active — детектируем моргание
    ear = compute_ear(image)
    if ear is None:
        session.frames_processed += 1
        if session.frames_processed >= MAX_FRAMES_FOR_BLINK:
            return {"status": "failed", "reason": "no_face_detected"}
        return {"status": "pending", "reason": "no_face_in_frame",
                "frames": session.frames_processed}

    blinked = update_blink_state(session, ear)

    if blinked and session.passive_passed:
        return {
            "status": "success",
            "ear_history": [round(v, 3) for v in session.ear_history],
            "frames": session.frames_processed,
        }

    if session.frames_processed >= MAX_FRAMES_FOR_BLINK:
        return {"status": "failed", "reason": "no_blink_detected_in_time"}

    return {
        "status": "pending",
        "ear": round(ear, 3),
        "frames": session.frames_processed,
        "blink_detected": session.blink_detected,
    }
