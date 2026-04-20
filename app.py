"""
FastAPI Backend — 2FA система с Face Recognition + OTP
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
from database import get_otp_secret, save_otp_secret, log_auth, download_model_if_needed
import pyotp
import qrcode
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from verifier import FaceVerifier
from database import (get_otp_secret, save_otp_secret, log_auth,
                       download_model_if_needed,
                       create_session, get_session,
                       update_session_attempts, delete_session)
# ─── Инициализация ──────────────────────────────────────────────────────────

app = FastAPI(title="2FA Face + OTP System", version="1.0.0")
templates = Jinja2Templates(directory="templates")

# Монтирование статических файлов
os.makedirs("static", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/users", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Глобальный объект верификатора (загружается при наличии модели)
verifier: Optional[FaceVerifier] = None

def load_verifier():
    global verifier
    model_path = "models/best_model.pth"
    # Если модели нет — скачиваем из Supabase Storage
    download_model_if_needed(model_path)
    if os.path.exists(model_path):
        try:
            verifier = FaceVerifier(model_path, "models/embeddings.json")
            print("✓ Верификатор загружен")
        except Exception as e:
            print(f"✗ Ошибка загрузки модели: {e}")


load_verifier()

# ─── OTP сессии ─────────────────────────────────────────────────────────────

# Временное хранилище OTP сессий (в production - Redis/DB)
otp_secrets: dict = {}     # {user_id: totp_secret}

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
    secret = get_otp_secret(user_id)   # из Supabase
    if not secret:
        secret = pyotp.random_base32()
        save_otp_secret(user_id, secret)  # в Supabase
    return secret


# ─── Вспомогательные функции ────────────────────────────────────────────────

def image_from_upload(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert('RGB')

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


# ─── HTML страницы ───────────────────────────────────────────────────────────

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
    # Показывает текущую структуру данных
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
    from database import get_all_users, get_embedding, get_otp_secret
    import json

    # Получаем список пользователей с доп. инфо
    user_ids = verifier.get_registered_users() if verifier else []
    users = []
    for uid in user_ids:
        has_otp = get_otp_secret(uid) is not None
        # Получаем дату из Supabase если есть
        try:
            from database import db
            res = db().table("user_embeddings").select("created_at").eq("user_id", uid).execute()
            created_at = res.data[0]["created_at"] if res.data else None
        except:
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
            # Сохраняем в стандартном размере
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

@app.delete("/api/admin/delete-user/{user_id}")
async def admin_delete_user(user_id: str):
    import shutil
    # Удаляем локальные данные
    user_dir = Path(f'data/users/{user_id}')
    if user_dir.exists():
        shutil.rmtree(user_dir)
    # Удаляем из verifier (память)
    if verifier:
        verifier.delete_user(user_id)
    # Удаляем из Supabase
    from database import delete_user_embedding
    delete_user_embedding(user_id)
    return {"success": True, "message": f"Пользователь {user_id} удалён"}


@app.get("/api/admin/user-stats/{user_id}")
async def admin_user_stats(user_id: str):
    try:
        from database import db
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
    except Exception as e:
        return {"user_id": user_id, "total": 0, "success_count": 0, "avg_similarity": None}


# ── API: Логи аутентификации ──────────────────────────────────────────────────


@app.get("/api/admin/logs")
async def admin_logs(filter: str = "all", limit: int = 50):
    try:
        from database import db
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


# ─── API: Обучение модели ────────────────────────────────────────────────────

@app.post("/api/train")
async def start_training(epochs: int = Form(30)):
    """Запускает обучение модели (синхронно для простоты)."""
    import subprocess
    import sys

    data_dir = Path('data/users')
    user_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if len(user_dirs) < 2:
        raise HTTPException(400, "Нужно минимум 2 пользователя с фото")

    try:
        result = subprocess.run(
            [sys.executable, 'train.py',
             '--data_dir', 'data/users',
             '--epochs', str(epochs),
             '--save_dir', 'models'],
            capture_output=True, text=True, timeout=600
        )

        if result.returncode == 0:
            load_verifier()
            try:
                from sync_static import sync_static
                sync_static()
            except Exception:
                pass
            # Перезагружаем модель
            return {
                "success": True,
                "message": "Обучение завершено!",
                "output": result.stdout[-2000:]  # Последние 2000 символов
            }
        else:
            return {
                "success": False,
                "error": result.stderr[-1000:]
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Превышено время обучения (10 минут)"}


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
        # Создаём OTP секрет для пользователя
        secret = get_or_create_otp_secret(user_id)
        totp = pyotp.TOTP(secret)

        # Генерируем QR-код
        uri = totp.provisioning_uri(name=user_id, issuer_name="2FA Healthcare")
        qr = qrcode.make(uri)
        buf = io.BytesIO()
        qr.save(buf, format='PNG')
        qr_b64 = base64.b64encode(buf.getvalue()).decode()

        result['otp_secret'] = secret
        result['qr_code'] = qr_b64

    return result


# ─── API: Верификация (2FA) ──────────────────────────────────────────────────

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
        expires_at = time.time() + 300
        create_session(session_id, user_id, expires_at, result['similarity'])
        print(f"✓ Сессия создана: {session_id[:20]}...")  # ← добавьте эту строку

        # Сохраняем сессию в Supabase (переживёт перезапуск)
        create_session(session_id, user_id, expires_at, result['similarity'])

        result['session_id'] = session_id
        result['message'] = 'Лицо верифицировано! Введите OTP.'
    else:
        log_auth(user_id, result['similarity'], False, False, False)
        result['message'] = f'Верификация не пройдена (сходство: {result["similarity"]:.3f})'

    return result


@app.post("/api/verify-otp")
async def verify_otp(
    session_id: str = Form(...),
    otp_code: str = Form(...)
):
    # Читаем сессию из Supabase
    session = get_session(session_id)
    print(f"Ищу сессию: {session_id[:20]}... → {session}")  # ← добавьте

    if not session:
        raise HTTPException(400, "Сессия не найдена или истекла")

    if time.time() > session['expires_at']:
        delete_session(session_id)
        raise HTTPException(400, "Сессия истекла. Начните заново.")

    if session['attempts'] >= 3:
        delete_session(session_id)
        return {
            'authenticated': False,
            'error': 'Превышено число попыток (3). Начните заново.'
        }

    user_id = session['user_id']
    secret = get_otp_secret(user_id)

    if not secret:
        raise HTTPException(400, f"OTP не настроен для {user_id}")

    totp = pyotp.TOTP(secret)
    valid = totp.verify(otp_code, valid_window=1)

    new_attempts = session['attempts'] + 1
    update_session_attempts(session_id, new_attempts)

    if valid:
        delete_session(session_id)
        log_auth(user_id, session['face_similarity'], True, True, True)
        return {
            'authenticated': True,
            'user_id': user_id,
            'face_similarity': session['face_similarity'],
            'message': f'✓ Аутентификация успешна! Добро пожаловать, {user_id}!',
            'timestamp': datetime.now().isoformat()
        }
    else:
        attempts_left = 3 - new_attempts
        if attempts_left <= 0:
            delete_session(session_id)
            log_auth(user_id, session['face_similarity'], True, False, False)
        return {
            'authenticated': False,
            'attempts_left': attempts_left,
            'message': f'Неверный OTP. Осталось попыток: {attempts_left}'
        }


# ─── API: Статистика ─────────────────────────────────────────────────────────

@app.get("/api/model-info")
async def model_info():
    """Информация о текущей модели и результатах обучения."""
    info = {
        "model_loaded": verifier is not None,
        "registered_users": verifier.get_registered_users() if verifier else [],
        "threshold": verifier.threshold if verifier else DEFAULT_THRESHOLD
    }

    # История обучения
    history = get_training_history()
    if history:
        info['training'] = {
            'epochs': len(history['loss']),
            'final_loss': round(history['loss'][-1], 4),
            'best_accuracy': round(max(history['accuracy']), 1),
            'final_accuracy': round(history['accuracy'][-1], 1)
        }

    # Информация о checkpoint
    checkpoint_path = 'models/best_model.pth'
    if os.path.exists(checkpoint_path):
        import torch
        cp = torch.load(checkpoint_path, map_location='cpu')
        info['model_details'] = {
            'best_epoch': cp.get('epoch'),
            'accuracy': round(cp.get('accuracy', 0), 1),
            'embedding_dim': cp.get('embedding_dim'),
            'trained_users': cp.get('users', [])
        }

    return info

DEFAULT_THRESHOLD = 0.70

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)
