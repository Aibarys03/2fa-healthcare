"""Запустить ОДИН РАЗ — переносит пользователей из файла в Supabase."""
import json, pyotp
from database import save_embedding, save_otp_secret, get_otp_secret

with open("models/embeddings.json", "r") as f:
    embeddings = json.load(f)

print(f"Найдено пользователей в файле: {len(embeddings)}")

for user_id, embedding in embeddings.items():
    save_embedding(user_id, embedding)
    print(f"✓ {user_id} — эмбеддинг перенесён")

    if not get_otp_secret(user_id):
        save_otp_secret(user_id, pyotp.random_base32())
        print(f"  + OTP создан")

print("\n✓ Готово! Проверьте Supabase → Table Editor → user_embeddings")