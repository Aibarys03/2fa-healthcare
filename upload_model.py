"""
Запустить ОДИН РАЗ локально для загрузки модели в Supabase Storage.
После загрузки этот файл больше не нужен.
"""
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

sb = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

print("Загружаю best_model.pth в Supabase Storage...")
with open("models/best_model.pth", "rb") as f:
    data = f.read()

result = sb.storage.from_("models").upload(
    "best_model.pth",
    data,
    {"content-type": "application/octet-stream", "upsert": "true"}
)
print("✓ Модель успешно загружена в облако!")
