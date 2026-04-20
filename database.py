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
