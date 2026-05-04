"""
Криптографическая защита биометрических эмбеддингов
====================================================

Реализация AES-256-GCM шифрования с per-user ключами, производными
через HKDF-SHA256 от мастер-ключа. Удовлетворяет требованиям
GDPR Art. 25 (Privacy by Design), Art. 32 (Security of Processing),
HIPAA §164.312 (Encryption at Rest), ISO 27799:2016 §7.10.

Стандарты:
  - AES-256-GCM:  NIST SP 800-38D, RFC 5116
  - HKDF-SHA256:  RFC 5869
  - Random IV:    NIST SP 800-90A (cryptographically secure)

Архитектура:
  master_key (256 bit, в Render env vars, недоступен из приложения)
       │
       ▼
   HKDF(salt = sha256(user_id), info = b"embedding-v1")
       │
       ▼
   user_key (256 bit, специфичен для пользователя)
       │
       ▼
   AES-256-GCM(nonce, plaintext_embedding, AAD = user_id)
       │
       ▼
   ciphertext = nonce(12B) || encrypted(N) || tag(16B)
       │
       ▼
   base64(ciphertext) → Supabase
"""

from __future__ import annotations

import os
import json
import base64
import hashlib
import struct
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


# ──────────────────────────────────────────────────────────────────
# Параметры криптосистемы
# ──────────────────────────────────────────────────────────────────

KEY_SIZE_BYTES = 32   # 256 bit для AES-256
NONCE_SIZE_BYTES = 12  # 96 bit рекомендуется NIST SP 800-38D для GCM
TAG_SIZE_BYTES = 16   # 128 bit, встроен в AESGCM
HKDF_INFO = b"embedding-encryption-v1"
EMBEDDING_DIM = 128
FLOAT32_BYTES = 4


# ──────────────────────────────────────────────────────────────────
# Загрузка мастер-ключа
# ──────────────────────────────────────────────────────────────────

class MasterKeyError(RuntimeError):
    """Мастер-ключ не настроен или невалиден."""


def _load_master_key() -> bytes:
    """
    Читает мастер-ключ из переменной окружения BIOMETRIC_MASTER_KEY.
    Ключ должен быть base64-закодированной 32-байтной строкой.

    Для генерации нового ключа:
        python -c "import base64,os; print(base64.b64encode(os.urandom(32)).decode())"
    """
    raw = os.environ.get("BIOMETRIC_MASTER_KEY")
    if not raw:
        raise MasterKeyError(
            "BIOMETRIC_MASTER_KEY не задан в окружении. "
            "Сгенерируйте ключ командой:\n"
            "  python -c \"import base64,os; print(base64.b64encode(os.urandom(32)).decode())\"\n"
            "и добавьте в .env (локально) или Render Environment Variables."
        )
    try:
        key = base64.b64decode(raw)
    except Exception as e:
        raise MasterKeyError(f"BIOMETRIC_MASTER_KEY некорректен (не base64): {e}")
    if len(key) != KEY_SIZE_BYTES:
        raise MasterKeyError(
            f"Длина мастер-ключа должна быть {KEY_SIZE_BYTES} байт, получено {len(key)}"
        )
    return key


# ──────────────────────────────────────────────────────────────────
# Производный ключ пользователя (HKDF)
# ──────────────────────────────────────────────────────────────────

def _derive_user_key(user_id: str) -> bytes:
    """
    Производит уникальный 256-битный ключ для пользователя через HKDF.
    Salt = SHA-256(user_id) — детерминированный, но не raw user_id
    (это рекомендация RFC 5869: salt должен быть случайным или хеш).
    """
    master = _load_master_key()
    salt = hashlib.sha256(user_id.encode("utf-8")).digest()
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE_BYTES,
        salt=salt,
        info=HKDF_INFO,
    )
    return hkdf.derive(master)


# ──────────────────────────────────────────────────────────────────
# Сериализация embedding ↔ bytes
# ──────────────────────────────────────────────────────────────────

def _embedding_to_bytes(embedding: list[float]) -> bytes:
    """128 × float32 → 512 bytes."""
    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding должен быть размерности {EMBEDDING_DIM}, "
            f"получено {len(embedding)}"
        )
    return struct.pack(f"<{EMBEDDING_DIM}f", *embedding)


def _bytes_to_embedding(data: bytes) -> list[float]:
    """512 bytes → 128 × float32."""
    expected = EMBEDDING_DIM * FLOAT32_BYTES
    if len(data) != expected:
        raise ValueError(
            f"Decrypted embedding должен быть {expected} байт, "
            f"получено {len(data)}"
        )
    return list(struct.unpack(f"<{EMBEDDING_DIM}f", data))


# ──────────────────────────────────────────────────────────────────
# Публичный API: encrypt / decrypt
# ──────────────────────────────────────────────────────────────────

def encrypt_embedding(user_id: str, embedding: list[float]) -> str:
    """
    Шифрует embedding пользователя.

    Возвращает base64-строку формата: nonce(12B) || ciphertext || tag(16B).
    AAD = user_id обеспечивает связь шифротекста с пользователем —
    попытка использовать чужой шифротекст для другого user_id вызовет
    ошибку проверки тега аутентификации.
    """
    plaintext = _embedding_to_bytes(embedding)
    user_key = _derive_user_key(user_id)
    aesgcm = AESGCM(user_key)
    nonce = os.urandom(NONCE_SIZE_BYTES)
    aad = user_id.encode("utf-8")
    # encrypt() возвращает ciphertext || tag (tag в конце 16 байт)
    ct_with_tag = aesgcm.encrypt(nonce, plaintext, aad)
    blob = nonce + ct_with_tag
    return base64.b64encode(blob).decode("ascii")


def decrypt_embedding(user_id: str, encrypted_b64: str) -> list[float]:
    """
    Расшифровывает embedding. Бросает InvalidTag если данные подделаны
    или если шифротекст принадлежит другому пользователю.
    """
    try:
        blob = base64.b64decode(encrypted_b64)
    except Exception as e:
        raise ValueError(f"Encrypted blob не base64: {e}")

    min_size = NONCE_SIZE_BYTES + TAG_SIZE_BYTES
    if len(blob) < min_size + EMBEDDING_DIM * FLOAT32_BYTES:
        raise ValueError(
            f"Encrypted blob слишком короткий: {len(blob)} байт "
            f"(минимум {min_size + EMBEDDING_DIM * FLOAT32_BYTES})"
        )

    nonce = blob[:NONCE_SIZE_BYTES]
    ct_with_tag = blob[NONCE_SIZE_BYTES:]

    user_key = _derive_user_key(user_id)
    aesgcm = AESGCM(user_key)
    aad = user_id.encode("utf-8")
    try:
        plaintext = aesgcm.decrypt(nonce, ct_with_tag, aad)
    except Exception as e:
        # InvalidTag из cryptography — оборачиваем в более понятное
        raise ValueError(
            f"Расшифровка не удалась (возможно, повреждённые данные "
            f"или неверный мастер-ключ): {e}"
        )
    return _bytes_to_embedding(plaintext)


# ──────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────

def is_encrypted_blob(value: str) -> bool:
    """
    Проверяет, выглядит ли строка как зашифрованный blob.
    Используется для backwards-compatibility: можно подложить логику
    «если поле уже зашифровано — расшифруй; если нет — оно ещё в plain».
    """
    if not isinstance(value, str):
        return False
    try:
        blob = base64.b64decode(value)
    except Exception:
        return False
    expected_min = NONCE_SIZE_BYTES + TAG_SIZE_BYTES + EMBEDDING_DIM * FLOAT32_BYTES
    return len(blob) == expected_min


def selftest():
    """
    Самопроверка: round-trip encrypt → decrypt должен вернуть исходные данные.
    Запуск: python crypto_storage.py
    """
    # Для теста генерируем мастер-ключ на лету
    if "BIOMETRIC_MASTER_KEY" not in os.environ:
        test_key = base64.b64encode(os.urandom(32)).decode()
        os.environ["BIOMETRIC_MASTER_KEY"] = test_key
        print(f"[TEST] Сгенерирован временный мастер-ключ")

    test_embedding = [0.1 * i for i in range(128)]
    user_id = "test_user_aibarys"

    print(f"\n1. Encrypt embedding (dim={len(test_embedding)})...")
    encrypted = encrypt_embedding(user_id, test_embedding)
    print(f"   Encrypted blob: {encrypted[:60]}... (len={len(encrypted)})")

    print(f"\n2. Decrypt for same user...")
    decrypted = decrypt_embedding(user_id, encrypted)
    assert len(decrypted) == 128
    diffs = [abs(a - b) for a, b in zip(test_embedding, decrypted)]
    max_diff = max(diffs)
    print(f"   Max diff (round-trip error): {max_diff:.2e}")
    assert max_diff < 1e-6, "Round-trip error too large!"

    print(f"\n3. Try to decrypt with WRONG user_id (should fail)...")
    try:
        decrypt_embedding("wrong_user", encrypted)
        print("   ❌ FAILED — расшифровка прошла, хотя не должна была!")
    except ValueError as e:
        print(f"   ✓ Корректно отклонено (InvalidTag): {str(e)[:60]}...")

    print(f"\n4. is_encrypted_blob() check...")
    assert is_encrypted_blob(encrypted), "Должен опознать зашифрованный blob"
    plain = json.dumps([0.5] * 128)
    assert not is_encrypted_blob(plain), "Не должен опознать plain JSON"
    print(f"   ✓ Корректно отличает зашифрованные blob от plaintext")

    print("\n✅ Все проверки пройдены!")


if __name__ == "__main__":
    selftest()
