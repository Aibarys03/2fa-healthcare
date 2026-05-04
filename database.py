import os
from supabase import create_client
from dotenv import load_dotenv
import json
import logging

from crypto_storage import (
    encrypt_embedding, decrypt_embedding, is_encrypted_blob, MasterKeyError,
)
from audit_chain import (
    make_log_entry, get_last_chain_hash, verify_chain,
)

logger = logging.getLogger(__name__)
load_dotenv()

_client = None


def db():
    """Возвращает клиент Supabase (создаётся один раз)."""
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL и SUPABASE_KEY не заданы в .env"
            )
        _client = create_client(url, key)
    return _client


def get_otp_secret(user_id: str):
    """Получить OTP-секрет пользователя из БД."""
    try:
        res = db().table("otp_secrets") \
            .select("secret") \
            .eq("user_id", user_id) \
            .execute()
        return res.data[0]["secret"] if res.data else None
    except Exception as e:
        print(f"DB get_otp_secret error: {e}")
        return None


def save_otp_secret(user_id: str, secret: str):
    """Сохранить OTP-секрет пользователя в БД."""
    try:
        db().table("otp_secrets").upsert({
            "user_id": user_id,
            "secret": secret
        }).execute()
    except Exception as e:
        print(f"DB save_otp_secret error: {e}")


def log_auth(user_id: str, face_similarity: float,
             face_ok: bool, otp_ok: bool, success: bool) -> None:
    """
    Логирует попытку аутентификации в auth_logs с криптографически
    связной цепочкой хешей (tamper-evident).

    Каждая запись содержит:
      - prev_hash: chain_hash предыдущей записи
      - chain_hash: SHA-256(prev_hash || serialize(this_record))

    Любая модификация старой записи делает все последующие хеши
    недействительными при audit-проходе.
    """
    try:
        prev_hash = get_last_chain_hash(db())
        record = make_log_entry(
            user_id=user_id,
            face_similarity=face_similarity,
            face_passed=face_ok,
            otp_passed=otp_ok,
            success=success,
            prev_hash=prev_hash,
        )
        db().table("auth_logs").insert(record).execute()
    except Exception as e:
        # Не блокируем аутентификацию из-за проблем с логом
        logger.error(f"Не удалось записать audit-лог: {e}")


# ────────────────────────────────────────────────────────────────────
# НОВАЯ функция: миграция старых embeddings в зашифрованный формат
# ────────────────────────────────────────────────────────────────────

def migrate_plaintext_to_encrypted() -> dict:
    """
    Проходит по всем записям user_embeddings и перешифровывает те,
    что лежат в plaintext.

    Запускать ОДИН РАЗ после развёртывания шифрования.
    После этой функции все embeddings будут в зашифрованном виде.
    """
    try:
        all_users = db().table("user_embeddings").select(
            "user_id, embedding, is_encrypted"
        ).execute()
    except Exception as e:
        return {"success": False, "error": str(e)}

    migrated = []
    skipped = []
    failed = []

    for row in (all_users.data or []):
        user_id = row["user_id"]
        raw = row.get("embedding")
        already_encrypted = row.get("is_encrypted", False)

        if already_encrypted or is_encrypted_blob(raw):
            skipped.append(user_id)
            continue

        try:
            # Supabase может вернуть embedding либо как:
            #   - list[float]  (если колонка float8[] или нативный jsonb)
            #   - list[str]    (если float8[] и драйвер строки возвращает)
            #   - str          (если text/varchar с JSON внутри)
            if isinstance(raw, list):
                plaintext_emb = [float(x) for x in raw]
            elif isinstance(raw, str):
                plaintext_emb = json.loads(raw)
            else:
                raise ValueError(f"Неизвестный тип embedding: {type(raw)}")
            save_embedding(user_id, plaintext_emb)  # перешифрует
            migrated.append(user_id)
        except Exception as e:
            failed.append({"user_id": user_id, "error": str(e)})

    return {
        "success": True,
        "migrated": migrated,
        "skipped_already_encrypted": skipped,
        "failed": failed,
        "total_migrated": len(migrated),
    }


# ────────────────────────────────────────────────────────────────────
# НОВАЯ функция: проверка целостности audit-журнала
# ────────────────────────────────────────────────────────────────────

def verify_audit_chain(limit: int = 1000) -> dict:
    """
    Прогон по последним N записям auth_logs и проверка целостности
    цепочки хешей. Игнорирует старые записи без chain_hash (которые
    были созданы до миграции).
    """
    try:
        res = db().table("auth_logs") \
            .select("user_id, timestamp, face_similarity, "
                    "face_passed, otp_passed, success, "
                    "prev_hash, chain_hash") \
            .order("created_at", desc=False) \
            .limit(limit) \
            .execute()
    except Exception as e:
        return {"valid": False, "error": str(e), "records_checked": 0,
                "broken_at": [], "details": [f"DB error: {e}"]}

    # Фильтруем только записи с заполненным chain_hash
    # (старые до миграции пропускаем)
    records = [r for r in (res.data or []) if r.get("chain_hash")]

    if not records:
        return {
            "valid": True,
            "records_checked": 0,
            "broken_at": [],
            "details": [],
            "note": "Нет записей с chain_hash. Пройдите полный flow аутентификации хотя бы раз."
        }

    return verify_chain(records)

def download_model_if_needed(model_path: str = "models/best_model.pth"):
    """Скачать модель из Supabase Storage если её нет локально."""
    if os.path.exists(model_path):
        return True
    try:
        print("Модель не найдена локально, скачиваю из облака...")
        os.makedirs("models", exist_ok=True)
        data = db().storage.from_("models") \
            .download("best_model.pth")
        with open(model_path, "wb") as f:
            f.write(data)
        print("✓ Модель скачана из Supabase Storage")
        return True
    except Exception as e:
        print(f"✗ Не удалось скачать модель: {e}")
        return False


def get_embedding(user_id: str) -> list[float] | None:
    """
    Загружает embedding пользователя из Supabase, прозрачно расшифровывая.

    Backwards-compatible: умеет читать как зашифрованные, так и старые
    plaintext-записи. После первого save_embedding запись автоматически
    становится зашифрованной.
    """
    try:
        res = db().table("user_embeddings") \
            .select("embedding, is_encrypted") \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()
    except Exception as e:
        logger.error(f"Ошибка чтения embedding для {user_id}: {e}")
        return None

    if not res.data:
        return None

    row = res.data[0]
    raw = row.get("embedding")
    is_encrypted = row.get("is_encrypted", False)

    if raw is None:
        return None

    # Heuristic: если флаг is_encrypted=True ИЛИ blob выглядит как зашифрованный
    if is_encrypted or is_encrypted_blob(raw):
        try:
            return decrypt_embedding(user_id, raw)
        except (MasterKeyError, ValueError) as e:
            logger.error(f"Не удалось расшифровать embedding {user_id}: {e}")
            return None

    # Иначе считаем plaintext (старый формат, до миграции)
    if isinstance(raw, list):
        return [float(x) for x in raw]
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Embedding {user_id} в неизвестном формате: {e}")
        return None

def save_embedding(user_id: str, embedding: list[float]) -> None:
    """
    Сохраняет embedding пользователя в Supabase в зашифрованном виде.

    AES-256-GCM с per-user ключом, производным через HKDF-SHA256
    от мастер-ключа BIOMETRIC_MASTER_KEY (Render env vars).

    Если мастер-ключ не настроен — сохраняем plaintext (с WARNING).
    Это обеспечивает обратную совместимость для разработки.
    """
    try:
        encrypted = encrypt_embedding(user_id, embedding)
        payload = {
            "user_id": user_id,
            "embedding": encrypted,  # зашифрованный blob
            "is_encrypted": True,  # флаг для миграции
        }
    except MasterKeyError as e:
        logger.warning(
            f"BIOMETRIC_MASTER_KEY не настроен — embedding для {user_id} "
            f"сохраняется в plaintext! Это небезопасно для production. "
            f"Детали: {e}"
        )
        payload = {
            "user_id": user_id,
            "embedding": json.dumps(embedding),
            "is_encrypted": False,
        }

    try:
        # Upsert: создаёт новую запись или обновляет существующую
        db().table("user_embeddings").upsert(payload, on_conflict="user_id").execute()
    except Exception as e:
        logger.error(f"Не удалось сохранить embedding для {user_id}: {e}")
        raise

def get_all_users() -> list:
    """Получить список всех зарегистрированных пользователей."""
    try:
        res = db().table("user_embeddings").select("user_id").execute()
        return [row["user_id"] for row in res.data]
    except Exception as e:
        print(f"DB get_all_users error: {e}")
        return []

def delete_user_embedding(user_id: str):
    """Удалить пользователя из БД."""
    try:
        db().table("user_embeddings").delete().eq("user_id", user_id).execute()
        db().table("otp_secrets").delete().eq("user_id", user_id).execute()
    except Exception as e:
        print(f"DB delete_user error: {e}")

def create_session(session_id: str, user_id: str, expires_at: float, similarity: float):
    try:
        db().table("otp_sessions").upsert({   # ← было insert, стало upsert
            "session_id":      session_id,
            "user_id":         user_id,
            "expires_at":      expires_at,
            "attempts":        0,
            "face_similarity": similarity
        }).execute()
    except Exception as e:
        print(f"DB create_session error: {e}")

def get_session(session_id: str):
    """Получить сессию из БД."""
    try:
        res = db().table("otp_sessions") \
            .select("*") \
            .eq("session_id", session_id) \
            .execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"DB get_session error: {e}")
        return None

def update_session_attempts(session_id: str, attempts: int):
    """Обновить счётчик попыток."""
    try:
        db().table("otp_sessions") \
            .update({"attempts": attempts}) \
            .eq("session_id", session_id) \
            .execute()
    except Exception as e:
        print(f"DB update_session error: {e}")

def delete_session(session_id: str):
    """Удалить сессию после использования."""
    try:
        db().table("otp_sessions") \
            .delete() \
            .eq("session_id", session_id) \
            .execute()
    except Exception as e:
        print(f"DB delete_session error: {e}")