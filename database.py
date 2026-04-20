import os
from supabase import create_client
from dotenv import load_dotenv

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


def log_auth(user_id: str, similarity: float,
             face_ok: bool, otp_ok: bool, success: bool):
    """Записать попытку аутентификации в лог."""
    try:
        db().table("auth_logs").insert({
            "user_id":         user_id,
            "face_similarity":  round(float(similarity), 4),
            "face_passed":      face_ok,
            "otp_passed":       otp_ok,
            "success":          success
        }).execute()
    except Exception as e:
        print(f"DB log_auth error: {e}")


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
def get_embedding(user_id: str):
    """Получить эмбеддинг пользователя из БД."""
    try:
        res = db().table("user_embeddings") \
            .select("embedding") \
            .eq("user_id", user_id) \
            .execute()
        return res.data[0]["embedding"] if res.data else None
    except Exception as e:
        print(f"DB get_embedding error: {e}")
        return None

def save_embedding(user_id: str, embedding: list):
    """Сохранить эмбеддинг пользователя в БД."""
    try:
        db().table("user_embeddings").upsert({
            "user_id":   user_id,
            "embedding": embedding
        }).execute()
    except Exception as e:
        print(f"DB save_embedding error: {e}")

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