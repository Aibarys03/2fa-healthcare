"""
FastAPI Backend — 2FA система с Face Recognition + OTP
Исправленная версия (v2):
  - Убран дубликат create_session
  - Убран дубликат импорта database
  - Robust обработка битых картинок (HTTP 400 вместо 500)
  - Endpoint /api/admin/regenerate-embeddings
  - Endpoint /api/admin/cleanup-sessions
  - DEFAULT_THRESHOLD перенесён в начало
  - Background task для долгого обучения
  - Чище verify_otp (явное разделение валидного/неверного OTP)
"""
import os
import io
import json
import time
import base64
import secrets
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import pyotp
import qrcode
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from verifier import FaceVerifier
from database import (
    get_otp_secret, save_otp_secret, log_auth,
    download_model_if_needed,
    create_session, get_session,
    update_session_attempts, delete_session,
    get_all_users, get_embedding as db_get_embedding,
    save_embedding, delete_user_embedding,
    db,
)
from liveness import (
    start_liveness_session,
    process_liveness_frame,
    end_liveness_session,
    get_liveness_session,
    reset_liveness_session,
)
from database import migrate_plaintext_to_encrypted, verify_audit_chain

# ─── Константы ──────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.70
TRAINING_TIMEOUT_SEC = 1800       # 30 минут — было 600 (10 минут)
SESSION_LIFETIME_SEC = 300        # 5 минут на ввод OTP
MAX_OTP_ATTEMPTS = 3


# ─── Инициализация ──────────────────────────────────────────────────────────
app = FastAPI(title="2FA Face + OTP System", version="2.0.0")
templates = Jinja2Templates(directory="templates")

os.makedirs("static", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/users", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

verifier: Optional[FaceVerifier] = None


def load_verifier():
    global verifier
    model_path = "models/best_model.pth"
    download_model_if_needed(model_path)

    if os.path.exists(model_path):
        try:
            verifier = FaceVerifier(model_path, "models/embeddings.json")
            print("✓ Верификатор загружен")
        except Exception as e:
            print(f"✗ Ошибка загрузки модели: {e}")


load_verifier()


# ─── OTP secrets (локальный fallback) ──────────────────────────────────────
otp_secrets: dict = {}
OTP_SECRETS_FILE = 'models/otp_secrets.json'


def load_otp_secrets():
    if os.path.exists(OTP_SECRETS_FILE):
        with open(OTP_SECRETS_FILE, 'r') as f:
            otp_secrets.update(json.load(f))


def save_otp_secrets():
    with open(OTP_SECRETS_FILE, 'w') as f:
        json.dump(otp_secrets, f)


load_otp_secrets()


def get_or_create_otp_secret(user_id: str) -> str:
    secret = get_otp_secret(user_id)
    if not secret:
        secret = pyotp.random_base32()
        save_otp_secret(user_id, secret)
    return secret


# ─── Вспомогательные функции ────────────────────────────────────────────────

def image_from_upload(file_bytes: bytes) -> Image.Image:
    """Безопасное чтение картинки из upload. Бросает HTTP 400 при ошибке."""
    if not file_bytes:
        raise HTTPException(400, "Пустой файл")
    try:
        return Image.open(io.BytesIO(file_bytes)).convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(400, f"Не удалось распознать изображение: {e}")


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def get_training_history():
    path = 'models/training_history.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


# ─── HTML страницы ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    model_exists = os.path.exists('models/best_model.pth')
    users = verifier.get_registered_users() if verifier else []
    history = get_training_history()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_exists": model_exists,
        "registered_users": users,
        "user_count": len(users),
        "history": history
    })


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/verify", response_class=HTMLResponse)
async def verify_page(request: Request):
    users = verifier.get_registered_users() if verifier else []
    return templates.TemplateResponse("verify.html", {
        "request": request,
        "registered_users": users
    })


@app.get("/upload-data", response_class=HTMLResponse)
async def upload_data_page(request: Request):
    users_info = []
    data_dir = Path('data/users')
    if data_dir.exists():
        for user_dir in sorted(data_dir.iterdir()):
            if user_dir.is_dir():
                imgs = list(user_dir.glob('*.jpg')) + list(user_dir.glob('*.jpeg')) + \
                       list(user_dir.glob('*.png'))
                users_info.append({'name': user_dir.name, 'count': len(imgs)})
    return templates.TemplateResponse("upload_data.html", {
        "request": request,
        "users_info": users_info
    })


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    user_ids = verifier.get_registered_users() if verifier else []
    users = []
    for uid in user_ids:
        has_otp = get_otp_secret(uid) is not None
        try:
            res = db().table("user_embeddings").select("created_at").eq("user_id", uid).execute()
            created_at = res.data[0]["created_at"] if res.data else None
        except Exception:
            created_at = None
        users.append({"user_id": uid, "has_otp": has_otp, "created_at": created_at})

    history = get_training_history()
    threshold = verifier.threshold if verifier else DEFAULT_THRESHOLD
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "users": users,
        "model_exists": os.path.exists('models/best_model.pth'),
        "history": history,
        "threshold": threshold,
    })


# ─── API: Загрузка данных для обучения ──────────────────────────────────────

@app.post("/api/upload-training-data")
async def upload_training_data(
    user_id: str = Form(...),
    files: list[UploadFile] = File(...)
):
    """Загружает фото пользователя для обучения модели."""
    user_dir = Path(f'data/users/{user_id}')
    user_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    errors = []

    for f in files:
        try:
            content = await f.read()
            img = Image.open(io.BytesIO(content)).convert('RGB')
            img = img.resize((300, 300), Image.LANCZOS)

            filename = f"{user_id}_{saved+1:03d}.jpg"
            img.save(user_dir / filename, 'JPEG', quality=95)
            saved += 1
        except Exception as e:
            errors.append(str(e))

    return {
        "success": True,
        "user_id": user_id,
        "saved": saved,
        "total_in_folder": len(list(user_dir.glob('*.jpg'))),
        "errors": errors
    }


# ─── Admin: управление пользователями ───────────────────────────────────────

@app.delete("/api/admin/delete-user/{user_id}")
async def admin_delete_user(user_id: str):
    import shutil
    user_dir = Path(f'data/users/{user_id}')
    if user_dir.exists():
        shutil.rmtree(user_dir)

    if verifier:
        verifier.delete_user(user_id)

    delete_user_embedding(user_id)
    return {"success": True, "message": f"Пользователь {user_id} удалён"}


@app.get("/api/admin/user-stats/{user_id}")
async def admin_user_stats(user_id: str):
    try:
        res = db().table("auth_logs") \
            .select("success, face_similarity, face_passed, otp_passed") \
            .eq("user_id", user_id) \
            .execute()
        logs = res.data or []
        total = len(logs)
        success_count = sum(1 for l in logs if l.get("success"))
        sims = [l["face_similarity"] for l in logs if l.get("face_similarity") is not None]
        avg_sim = sum(sims) / len(sims) if sims else None
        return {
            "user_id": user_id,
            "total": total,
            "success_count": success_count,
            "avg_similarity": avg_sim,
        }
    except Exception:
        return {"user_id": user_id, "total": 0, "success_count": 0, "avg_similarity": None}


@app.get("/api/admin/logs")
async def admin_logs(filter: str = "all", limit: int = 50):
    try:
        query = db().table("auth_logs") \
            .select("*") \
            .order("created_at", desc=True) \
            .limit(limit)
        if filter == "success":
            query = query.eq("success", True)
        elif filter == "fail":
            query = query.eq("success", False)
        res = query.execute()
        return {"logs": res.data or []}
    except Exception as e:
        return {"logs": [], "error": str(e)}


@app.post("/api/admin/regenerate-embeddings")
async def admin_regenerate_embeddings():
    """
    Пересчитывает embeddings для всех пользователей в data/users/
    с использованием текущей CNN. Нужно вызывать после переобучения.
    """
    if not verifier:
        raise HTTPException(400, "Модель не загружена")

    data_dir = Path("data/users")
    if not data_dir.exists():
        raise HTTPException(400, "Папка data/users/ не найдена")

    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    processed = []
    failed = []

    for user_dir in sorted(data_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        images = [p for p in sorted(user_dir.iterdir()) if p.suffix.lower() in valid_ext]
        if not images:
            continue

        try:
            pil_images = [Image.open(p).convert("RGB") for p in images]
            result = verifier.register_user(user_dir.name, pil_images)
            if result.get("success"):
                processed.append({
                    "user_id": user_dir.name,
                    "photos_used": result.get("photos_used", len(images)),
                })
            else:
                failed.append({"user_id": user_dir.name, "error": result.get("error")})
        except Exception as e:
            failed.append({"user_id": user_dir.name, "error": str(e)})

    return {
        "success": True,
        "processed": processed,
        "failed": failed,
        "total_processed": len(processed),
        "total_failed": len(failed),
    }


@app.post("/api/admin/migrate-encryption")
async def admin_migrate_encryption():
    """
    Одноразовая миграция: перешифровывает все plaintext embeddings
    в формат AES-256-GCM. Запускать после первоначального развёртывания
    шифрования.

    После успешного запуска все embeddings в Supabase будут зашифрованы;
    повторный вызов безопасен (идемпотентен) — уже зашифрованные записи
    пропускаются.
    """
    result = migrate_plaintext_to_encrypted()
    if not result.get("success"):
        raise HTTPException(500, f"Migration failed: {result.get('error')}")
    return result


@app.get("/api/admin/verify-audit-chain")
async def admin_verify_audit_chain(limit: int = 1000):
    """
    Проверяет целостность tamper-evident audit-журнала.
    Возвращает:
      - valid: True если цепочка корректна
      - records_checked: сколько записей проверено
      - broken_at: индексы записей с обнаруженными нарушениями
      - details: текстовые описания нарушений

    Если valid=False — это означает, что кто-то изменил или удалил
    запись в auth_logs минуя API. Это критический инцидент безопасности.
    """
    result = verify_audit_chain(limit=limit)
    return result

@app.post("/api/admin/cleanup-sessions")
async def admin_cleanup_sessions():
    """
    Удаляет из Supabase все otp_sessions с expires_at в прошлом.
    Можно вызывать вручную или через cron каждые 10 минут.
    """
    try:
        now = time.time()
        res = db().table("otp_sessions") \
            .delete() \
            .lt("expires_at", now) \
            .execute()
        deleted_count = len(res.data) if res.data else 0
        return {"success": True, "deleted_sessions": deleted_count}
    except Exception as e:
        raise HTTPException(500, f"Ошибка очистки: {e}")


@app.get("/api/dataset-info")
async def dataset_info():
    """Информация о текущем датасете."""
    data_dir = Path('data/users')
    users = {}
    if data_dir.exists():
        for user_dir in sorted(data_dir.iterdir()):
            if user_dir.is_dir():
                imgs = list(user_dir.glob('*.jpg')) + list(user_dir.glob('*.jpeg')) + \
                       list(user_dir.glob('*.png'))
                users[user_dir.name] = len(imgs)
    return {
        "total_users": len(users),
        "users": users,
        "ready_for_training": len(users) >= 2 and all(v >= 2 for v in users.values())
    }


# ─── API: Обучение модели ───────────────────────────────────────────────────

# Глобальное состояние тренировки (для асинхронного запуска)
_training_state = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "success": None,
    "output_tail": None,
    "error": None,
}


def _do_training(epochs: int):
    """Запускается в background task — обучение на 30 минут максимум."""
    import subprocess
    import sys

    _training_state["running"] = True
    _training_state["started_at"] = time.time()
    _training_state["finished_at"] = None
    _training_state["success"] = None
    _training_state["output_tail"] = None
    _training_state["error"] = None

    try:
        result = subprocess.run(
            [sys.executable, 'train.py',
             '--data_dir', 'data/users',
             '--epochs', str(epochs),
             '--save_dir', 'models'],
            capture_output=True, text=True,
            timeout=TRAINING_TIMEOUT_SEC,
        )
        if result.returncode == 0:
            load_verifier()
            try:
                from sync_static import sync_static
                sync_static()
            except Exception:
                pass
            _training_state["success"] = True
            _training_state["output_tail"] = result.stdout[-2000:]
        else:
            _training_state["success"] = False
            _training_state["error"] = result.stderr[-1000:]
    except subprocess.TimeoutExpired:
        _training_state["success"] = False
        _training_state["error"] = f"Превышено время обучения ({TRAINING_TIMEOUT_SEC // 60} минут)"
    except Exception as e:
        _training_state["success"] = False
        _training_state["error"] = str(e)
    finally:
        _training_state["running"] = False
        _training_state["finished_at"] = time.time()


@app.post("/api/train")
async def start_training(background_tasks: BackgroundTasks, epochs: int = Form(30)):
    """
    Запускает обучение в фоне. Возвращает сразу.
    Прогресс/результат — через GET /api/train-status.
    """
    data_dir = Path('data/users')
    user_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if len(user_dirs) < 2:
        raise HTTPException(400, "Нужно минимум 2 пользователя с фото")

    if _training_state["running"]:
        return {"success": False, "error": "Обучение уже запущено"}

    background_tasks.add_task(_do_training, epochs)
    return {
        "success": True,
        "message": f"Обучение запущено в фоне ({epochs} эпох). "
                   f"Проверяйте /api/train-status каждые 30 секунд.",
    }


@app.get("/api/train-status")
async def train_status():
    """Текущий статус последнего обучения."""
    return {
        "running": _training_state["running"],
        "started_at": _training_state["started_at"],
        "finished_at": _training_state["finished_at"],
        "success": _training_state["success"],
        "error": _training_state["error"],
        "output_tail": _training_state["output_tail"],
    }


# ─── API: Регистрация пользователя ──────────────────────────────────────────

@app.post("/api/register")
async def register_user(
    user_id: str = Form(...),
    files: list[UploadFile] = File(...)
):
    """Регистрирует пользователя: вычисляет эмбеддинг из загруженных фото."""
    if not verifier:
        raise HTTPException(400, "Модель не загружена. Сначала обучите модель!")

    images = []
    for f in files:
        content = await f.read()
        images.append(image_from_upload(content))

    result = verifier.register_user(user_id, images)

    if result['success']:
        secret = get_or_create_otp_secret(user_id)
        totp = pyotp.TOTP(secret)

        uri = totp.provisioning_uri(name=user_id, issuer_name="2FA Healthcare")
        qr = qrcode.make(uri)
        buf = io.BytesIO()
        qr.save(buf, format='PNG')
        qr_b64 = base64.b64encode(buf.getvalue()).decode()

        result['otp_secret'] = secret
        result['qr_code'] = qr_b64

    return result


# ─── API: Верификация (2FA) ─────────────────────────────────────────────────

@app.post("/api/verify-face")
async def verify_face(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    if not verifier:
        raise HTTPException(400, "Модель не загружена")

    content = await file.read()
    probe_image = image_from_upload(content)

    result = verifier.verify(user_id, probe_image)

    if result.get('verified'):
        session_id = secrets.token_urlsafe(32)
        expires_at = time.time() + SESSION_LIFETIME_SEC

        # Создаём face-сессию в Supabase
        create_session(session_id, user_id, expires_at, result['similarity'])

        # Создаём liveness-сессию в памяти (отдельно от face-сессии)
        start_liveness_session(session_id)

        result['session_id'] = session_id
        # Указываем фронту что следующий шаг — liveness, а не сразу OTP
        result['next_step'] = 'liveness'
        result['message'] = 'Лицо распознано. Подтвердите, что вы живой человек — моргните.'
    else:
        log_auth(user_id, result.get('similarity', 0.0), False, False, False)
        sim = result.get("similarity", 0.0)
        result['message'] = f'Верификация не пройдена (сходство: {sim:.3f})'

    return result


# ─── Liveness Detection ─────────────────────────────────────────────────────

@app.post("/api/verify-liveness")
async def verify_liveness(
        session_id: str = Form(...),
        file: UploadFile = File(...)
):
    session = get_session(session_id)
    if not session:
        end_liveness_session(session_id)
        return {"status": "session_expired",
                "reason": "face_session_not_found"}

    if time.time() > session['expires_at']:
        delete_session(session_id)
        end_liveness_session(session_id)
        return {"status": "session_expired",
                "reason": "face_session_timeout"}

    content = await file.read()
    image = image_from_upload(content)
    result = process_liveness_frame(session_id, image)

    # ────────────────────────────────────────────────────────────
    # НОВОЕ: при успешном моргании выполняем ВТОРИЧНУЮ CNN-проверку
    # ────────────────────────────────────────────────────────────
    if result["status"] == "success":
        live_session = get_liveness_session(session_id)
        if live_session and live_session.last_open_frame_pil and verifier:
            user_id = session['user_id']
            try:
                # Повторно вериф embedding пользователя против кадра с камеры
                second_check = verifier.verify(
                    user_id, live_session.last_open_frame_pil
                )
                second_sim = second_check.get("similarity", 0.0)
                second_passed = second_check.get("verified", False)

                # Дополнительная информация в ответе
                result["second_cnn_check"] = {
                    "similarity": round(second_sim, 4),
                    "passed": bool(second_passed),
                }

                if not second_passed:
                    # Атака обнаружена: лицо в камере НЕ соответствует
                    # тому, что было загружено на шаге 1
                    log_auth(
                        user_id, second_sim,
                        face_ok=True,  # первая проверка прошла
                        otp_ok=False,
                        success=False,
                    )
                    delete_session(session_id)
                    end_liveness_session(session_id)
                    return {
                        "status": "failed",
                        "reason": "second_cnn_mismatch",
                        "second_cnn_check": result["second_cnn_check"],
                        "session_expired": True,
                        "message": (
                            "Лицо в момент моргания не соответствует "
                            "загруженному фото. Возможна попытка обхода "
                            "через чужое фото."
                        ),
                    }
            except Exception as e:
                # Если double-check не удался по техническим причинам —
                # логируем, но не блокируем (fail-open для UX);
                # альтернатива — fail-closed: result["status"] = "failed"
                logger.error(f"Second CNN check failed for {session_id}: {e}")

    if result["status"] == "failed":
        live = get_liveness_session(session_id)
        if live is None or getattr(live, "fail_count", 0) >= 2:
            log_auth(session['user_id'], session['face_similarity'],
                     face_ok=True, otp_ok=False, success=False)
            delete_session(session_id)
            end_liveness_session(session_id)
            result["session_expired"] = True
        else:
            if live is not None:
                live.fail_count = getattr(live, "fail_count", 0) + 1

    return result


@app.post("/api/reset-liveness")
async def reset_liveness(session_id: str = Form(...)):
    """
    Кнопка 'Попробовать ещё раз' с фронта.
    Обнуляет состояние liveness-FSM, не убивая face-сессию.
    """
    session = get_session(session_id)
    if not session:
        end_liveness_session(session_id)
        return {"status": "session_expired",
                "reason": "face_session_not_found"}

    if time.time() > session['expires_at']:
        delete_session(session_id)
        end_liveness_session(session_id)
        return {"status": "session_expired",
                "reason": "face_session_timeout"}

    reset_liveness_session(session_id)
    return {"status": "ok",
            "message": "Liveness reset, please blink again"}


@app.post("/api/verify-otp")
async def verify_otp(
    session_id: str = Form(...),
    otp_code: str = Form(...)
):
    session = get_session(session_id)
    if not session:
        raise HTTPException(400, "Сессия не найдена или истекла")

    if time.time() > session['expires_at']:
        delete_session(session_id)
        end_liveness_session(session_id)
        raise HTTPException(400, "Сессия истекла. Начните заново.")

    # Liveness обязательна перед вводом OTP
    live = get_liveness_session(session_id)
    if live is None or not live.blink_detected or not live.passive_passed:
        raise HTTPException(
            400,
            "Liveness-проверка не пройдена. Подтвердите живость прежде чем вводить OTP."
        )

    if session['attempts'] >= MAX_OTP_ATTEMPTS:
        delete_session(session_id)
        end_liveness_session(session_id)
        return {
            'authenticated': False,
            'error': f'Превышено число попыток ({MAX_OTP_ATTEMPTS}). Начните заново.'
        }

    user_id = session['user_id']
    secret = get_otp_secret(user_id)
    if not secret:
        raise HTTPException(400, f"OTP не настроен для {user_id}")

    totp = pyotp.TOTP(secret)
    valid = totp.verify(otp_code, valid_window=1)

    new_attempts = session['attempts'] + 1
    update_session_attempts(session_id, new_attempts)

    # Логируем результат до удаления сессии
    log_auth(user_id, session['face_similarity'], True, valid, valid)

    if valid:
        delete_session(session_id)
        end_liveness_session(session_id)
        return {
            'authenticated': True,
            'user_id': user_id,
            'face_similarity': session['face_similarity'],
            'message': f'✓ Аутентификация успешна! Добро пожаловать, {user_id}!',
            'timestamp': datetime.now().isoformat()
        }

    # Неверный OTP
    attempts_left = MAX_OTP_ATTEMPTS - new_attempts
    if attempts_left <= 0:
        delete_session(session_id)
        end_liveness_session(session_id)

    return {
        'authenticated': False,
        'attempts_left': max(0, attempts_left),
        'message': f'Неверный OTP. Осталось попыток: {max(0, attempts_left)}'
    }


# ─── API: Статистика ────────────────────────────────────────────────────────

@app.get("/api/model-info")
async def model_info():
    """Информация о текущей модели и результатах обучения."""
    info = {
        "model_loaded": verifier is not None,
        "registered_users": verifier.get_registered_users() if verifier else [],
        "threshold": verifier.threshold if verifier else DEFAULT_THRESHOLD
    }

    history = get_training_history()
    if history:
        info['training'] = {
            'epochs': len(history.get('loss', [])),
            'final_loss': round(history['loss'][-1], 4) if history.get('loss') else None,
            'best_accuracy': round(max(history['accuracy']), 1) if history.get('accuracy') else None,
            'final_accuracy': round(history['accuracy'][-1], 1) if history.get('accuracy') else None,
        }
        if 'final_metrics' in history:
            fm = history['final_metrics']
            info['final_metrics'] = {
                'far': round(fm.get('far', 0), 2),
                'frr': round(fm.get('frr', 0), 2),
                'eer': round(fm.get('eer', 0), 2),
                'roc_auc': round(fm.get('roc_auc', 0), 4),
                'optimal_threshold': round(fm.get('optimal_threshold', 0.7), 3),
            }

    checkpoint_path = 'models/best_model.pth'
    if os.path.exists(checkpoint_path):
        import torch
        cp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        info['model_details'] = {
            'best_epoch': cp.get('epoch'),
            'accuracy': round(cp.get('accuracy', 0), 1),
            'embedding_dim': cp.get('embedding_dim'),
            'trained_users': cp.get('users', [])
        }

    return info


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)
