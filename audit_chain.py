"""
Tamper-Evident Audit Logging via SHA-256 Hash Chain
====================================================

Реализация append-only журнала аутентификации с криптографически
связной цепочкой. Любая модификация ранее записанной записи делает
все последующие хеши недействительными — это обнаруживается при
audit-проходе.

Стандарты:
  - SHA-256:               NIST FIPS 180-4
  - Audit logging:         ISO 27799:2016 §7.10 (Health informatics)
  - HIPAA §164.312(b):     Audit controls
  - GDPR Art. 5(1)(f):     Integrity and confidentiality

Архитектура:
  L_n = (timestamp, user_id, face_sim, face_passed, otp_passed, success, prev_hash)
  L_n.chain_hash = SHA-256(prev_hash || canonical_serialize(L_n))

  При вставке:    chain_hash вычисляется и сохраняется в записи
  При аудите:     прогон по всему журналу — пересчёт хешей и сравнение
                  Любое расхождение → запись была изменена/удалена
"""

from __future__ import annotations

import json
import hashlib
import time
from typing import Optional, Iterable


# Префикс для самой первой записи (нет prev_hash) — фиксирован
GENESIS_PREV_HASH = "0" * 64


def _canonical_serialize(record: dict) -> str:
    """
    Канонизирует запись в детерминированный JSON для хеширования.
    Удаляет служебные поля (id, chain_hash, created_at), сортирует ключи.
    """
    EXCLUDE = {"id", "chain_hash", "created_at"}
    cleaned = {k: v for k, v in record.items() if k not in EXCLUDE}
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def compute_chain_hash(record: dict, prev_hash: str) -> str:
    """
    Считает chain_hash для одной записи журнала.

    chain_hash = SHA-256(prev_hash || canonical_serialize(record))

    prev_hash включается в хеш — это и есть «цепочка»: чтобы подделать
    запись N, нужно пересчитать все записи N+1, N+2, ... до конца.
    """
    payload = prev_hash + _canonical_serialize(record)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_last_chain_hash(supabase_client) -> str:
    """
    Возвращает chain_hash последней записи в auth_logs.
    Если журнал пуст — возвращает GENESIS_PREV_HASH.
    """
    try:
        res = supabase_client.table("auth_logs") \
            .select("chain_hash") \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        if res.data and res.data[0].get("chain_hash"):
            return res.data[0]["chain_hash"]
    except Exception as e:
        # Если поле chain_hash ещё не добавлено в схему — fallback на genesis
        # (одноразово после миграции)
        print(f"[audit_chain] last hash lookup failed: {e}")
    return GENESIS_PREV_HASH


def make_log_entry(user_id: str, face_similarity: float,
                   face_passed: bool, otp_passed: bool, success: bool,
                   prev_hash: str) -> dict:
    """
    Формирует запись журнала с уже посчитанным chain_hash.
    Возвращает dict, готовый для INSERT в Supabase.

    timestamp фиксируется как секунды Unix (целое число) — это убирает
    проблемы с разной точностью времени между серверами.
    """
    record = {
        "user_id": user_id,
        "timestamp": int(time.time()),
        "face_similarity": float(face_similarity) if face_similarity is not None else None,
        "face_passed": bool(face_passed),
        "otp_passed": bool(otp_passed),
        "success": bool(success),
        "prev_hash": prev_hash,
    }
    record["chain_hash"] = compute_chain_hash(record, prev_hash)
    return record


# ──────────────────────────────────────────────────────────────────
# Аудит — проверка целостности журнала
# ──────────────────────────────────────────────────────────────────

def verify_chain(records: Iterable[dict]) -> dict:
    """
    Проверяет целостность последовательности записей.
    records — итерируемая в хронологическом порядке (от старых к новым).

    Возвращает dict:
      {
        "valid": bool,                 общая валидность цепочки
        "records_checked": int,        сколько записей проверено
        "broken_at": list[int],        индексы записей с разрывом
        "details": list[str],          описания нарушений
      }
    """
    result = {
        "valid": True,
        "records_checked": 0,
        "broken_at": [],
        "details": [],
    }
    prev_hash = GENESIS_PREV_HASH

    for idx, rec in enumerate(records):
        result["records_checked"] += 1

        # Проверка 1: prev_hash в записи должен совпадать с тем, который мы ожидаем
        rec_prev = rec.get("prev_hash", "")
        if rec_prev != prev_hash:
            result["valid"] = False
            result["broken_at"].append(idx)
            result["details"].append(
                f"Запись #{idx}: prev_hash в БД = {rec_prev[:16]}..., "
                f"ожидался {prev_hash[:16]}... (предыдущая запись изменена или удалена)"
            )

        # Проверка 2: пересчитанный chain_hash должен совпадать с сохранённым
        expected_hash = compute_chain_hash(rec, rec_prev)
        stored_hash = rec.get("chain_hash", "")
        if stored_hash != expected_hash:
            result["valid"] = False
            result["broken_at"].append(idx)
            result["details"].append(
                f"Запись #{idx}: chain_hash в БД = {stored_hash[:16]}..., "
                f"пересчитанный = {expected_hash[:16]}... (содержимое записи модифицировано)"
            )

        prev_hash = rec.get("chain_hash", "") or expected_hash

    return result


# ──────────────────────────────────────────────────────────────────
# Selftest
# ──────────────────────────────────────────────────────────────────

def selftest():
    """
    Самопроверка: построить цепочку, проверить, потом «подделать»
    среднюю запись и убедиться что аудит это ловит.
    """
    print("1. Строю цепочку из 5 записей...")
    chain = []
    prev = GENESIS_PREV_HASH
    for i in range(5):
        rec = make_log_entry(
            user_id=f"user_{i}",
            face_similarity=0.85 + i * 0.01,
            face_passed=True,
            otp_passed=(i % 2 == 0),
            success=(i % 2 == 0),
            prev_hash=prev,
        )
        chain.append(rec)
        prev = rec["chain_hash"]
        print(f"   Record #{i}: chain_hash = {rec['chain_hash'][:16]}...")

    print("\n2. Проверяю валидную цепочку...")
    audit = verify_chain(chain)
    assert audit["valid"], "Свежая цепочка должна быть валидной!"
    print(f"   ✓ valid=True, records_checked={audit['records_checked']}")

    print("\n3. Подделываю запись #2 (меняю user_id)...")
    chain[2]["user_id"] = "ATTACKER"
    audit = verify_chain(chain)
    assert not audit["valid"], "Подделка должна обнаруживаться!"
    print(f"   ✓ valid=False, broken_at={audit['broken_at']}")
    for d in audit["details"]:
        print(f"   - {d}")

    print("\n4. Чиню запись #2 (восстанавливаю как было)...")
    chain[2]["user_id"] = "user_2"
    audit = verify_chain(chain)
    assert audit["valid"], "После восстановления цепочка должна снова быть валидной!"
    print("   ✓ Цепочка снова валидна")

    print("\n5. Удаляю запись #3 (разрыв цепочки)...")
    chain_with_gap = chain[:3] + chain[4:]
    audit = verify_chain(chain_with_gap)
    assert not audit["valid"], "Удаление должно обнаруживаться!"
    print(f"   ✓ valid=False, broken_at={audit['broken_at']}")

    print("\n✅ Все проверки tamper-evident логирования пройдены!")


if __name__ == "__main__":
    selftest()
