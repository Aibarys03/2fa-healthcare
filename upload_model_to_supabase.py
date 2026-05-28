"""
Загрузка обученной модели и training_history.json в Supabase Storage.


Запускать ПОСЛЕ переобучения модели (когда обновился models/best_model.pth):

    python upload_model_to_supabase.py

Что делает:
  1. Читает models/best_model.pth + models/training_history.json
  2. Загружает оба в Supabase Storage с перезаписью существующих
  3. Печатает сводку (размер файлов, число пользователей в модели)

После этого Render при следующем запуске (или ручном Manual Deploy)
скачает обновлённую модель через download_model_if_needed().

Требования:
  - .env с SUPABASE_URL и SUPABASE_KEY (service_role или anon с правами на storage)
  - Существующий bucket в Supabase Storage (имя задаётся в BUCKET_NAME ниже)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client

# ─── Настройки ─────────────────────────────────────────────────────
BUCKET_NAME = "models"          # имя bucket в Supabase Storage
MODEL_FILE = "models/best_model.pth"
HISTORY_FILE = "models/training_history.json"

# Имена файлов в bucket (как они там лежат)
MODEL_KEY = "best_model.pth"
HISTORY_KEY = "training_history.json"


def main():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("❌ SUPABASE_URL или SUPABASE_KEY не заданы в .env")
        sys.exit(1)

    client = create_client(url, key)
    storage = client.storage.from_(BUCKET_NAME)

    # Проверка локальных файлов
    for f in (MODEL_FILE, HISTORY_FILE):
        if not Path(f).exists():
            print(f"❌ Не найден: {f}")
            sys.exit(1)

    # Информация о модели
    try:
        import torch
        cp = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
        users = cp.get("users", [])
        epoch = cp.get("epoch")
        acc = cp.get("accuracy")
        print(f"📊 Модель: epoch={epoch}, accuracy={acc:.1f}%, "
              f"{len(users)} пользователей: {users}")
    except Exception as e:
        print(f"⚠ Не удалось прочитать checkpoint: {e}")

    # Загрузка модели
    size_mb = Path(MODEL_FILE).stat().st_size / 1024 / 1024
    print(f"\n⬆ Загружаю {MODEL_FILE} ({size_mb:.1f} MB)...")
    with open(MODEL_FILE, "rb") as f:
        try:
            storage.upload(MODEL_KEY, f, {"upsert": "true",
                                         "content-type": "application/octet-stream"})
            print(f"   ✓ Загружено в {BUCKET_NAME}/{MODEL_KEY}")
        except Exception as e:
            # Fallback: некоторые версии клиента не поддерживают upsert через kwargs
            print(f"   upsert не сработал ({e}); пробую удалить + загрузить заново")
            try:
                storage.remove([MODEL_KEY])
            except Exception:
                pass
            f.seek(0)
            storage.upload(MODEL_KEY, f)
            print(f"   ✓ Загружено в {BUCKET_NAME}/{MODEL_KEY}")

    # Загрузка training_history
    size_kb = Path(HISTORY_FILE).stat().st_size / 1024
    print(f"\n⬆ Загружаю {HISTORY_FILE} ({size_kb:.1f} KB)...")
    with open(HISTORY_FILE, "rb") as f:
        try:
            storage.upload(HISTORY_KEY, f, {"upsert": "true",
                                            "content-type": "application/json"})
            print(f"   ✓ Загружено в {BUCKET_NAME}/{HISTORY_KEY}")
        except Exception as e:
            print(f"   upsert не сработал ({e}); пробую удалить + загрузить заново")
            try:
                storage.remove([HISTORY_KEY])
            except Exception:
                pass
            f.seek(0)
            storage.upload(HISTORY_KEY, f)
            print(f"   ✓ Загружено в {BUCKET_NAME}/{HISTORY_KEY}")

    print("\n✅ Готово! На Render выполните Manual Deploy → Clear build cache & deploy")
    print("   После этого новая модель и history будут активны.")


if __name__ == "__main__":
    main()
