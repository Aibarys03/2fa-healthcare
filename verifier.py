"""
Face Verification Engine — загрузка модели и верификация лиц
"""

import os
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

from model import FaceEmbeddingNet

IMG_SIZE = 100
EMBEDDING_DIM = 128
DEFAULT_THRESHOLD = 0.70


class FaceVerifier:
    """
    Движок верификации лиц на основе обученной CNN + triplet loss.
    Поддерживает:
    - Загрузку сохранённых эмбеддингов пользователей
    - Верификацию одного изображения против эталона
    - Добавление новых пользователей без переобучения
    """

    def __init__(self, model_path: str, embeddings_path: str = 'models/embeddings.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = DEFAULT_THRESHOLD
        self.embeddings_path = embeddings_path

        # Загрузка модели
        self.model = self._load_model(model_path)

        # Трансформация изображений
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Словарь эмбеддингов: {user_id: embedding_list}
        self.user_embeddings: dict = {}
        self._load_embeddings()

    def _load_model(self, model_path: str) -> FaceEmbeddingNet:
        """Загружает обученную модель из файла."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        emb_dim = checkpoint.get('embedding_dim', EMBEDDING_DIM)
        model = FaceEmbeddingNet(embedding_dim=emb_dim).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        self.threshold = checkpoint.get('threshold', DEFAULT_THRESHOLD)
        print(f"Модель загружена. Точность при обучении: {checkpoint.get('accuracy', 'N/A'):.1f}%")
        print(f"Порог верификации: {self.threshold}")
        return model

    def _load_embeddings(self):
        """Загружает эмбеддинги из Supabase (и из файла как fallback)."""
        from database import get_all_users, get_embedding as db_get_embedding

        # Сначала пробуем загрузить из Supabase
        try:
            users = get_all_users()
            if users:
                for user_id in users:
                    emb = db_get_embedding(user_id)
                    if emb:
                        self.user_embeddings[user_id] = torch.tensor(emb)
                print(f"Загружены эмбеддинги из Supabase: {len(self.user_embeddings)} пользователей")
                return
        except Exception as e:
            print(f"Supabase недоступен, пробую локальный файл: {e}")

        # Fallback — локальный файл (для разработки без интернета)
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'r') as f:
                data = json.load(f)
            self.user_embeddings = {
                uid: torch.tensor(embs) for uid, embs in data.items()
            }
            print(f"Загружены эмбеддинги из файла: {len(self.user_embeddings)} пользователей")

    def _save_embeddings(self):
        """Сохраняет эмбеддинги пользователей на диск."""
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        data = {
            uid: embs.tolist() for uid, embs in self.user_embeddings.items()
        }
        with open(self.embeddings_path, 'w') as f:
            json.dump(data, f)

    def get_embedding(self, image: Image.Image) -> torch.Tensor:
        """Вычисляет эмбеддинг для изображения."""
        img_tensor = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img_tensor)
        return embedding.squeeze(0).cpu()

    def register_user(self, user_id: str, images: list[Image.Image]) -> dict:
        """
        Регистрирует нового пользователя по набору фотографий.
        Создаёт усреднённый эталонный эмбеддинг.
        """
        if len(images) < 1:
            return {'success': False, 'error': 'Нужно хотя бы одно фото'}

        embeddings = []
        for img in images:
            emb = self.get_embedding(img)
            embeddings.append(emb)

        # Усредняем все эмбеддинги и нормализуем
        mean_emb = torch.stack(embeddings).mean(dim=0)
        mean_emb = F.normalize(mean_emb, p=2, dim=0)

        self.user_embeddings[user_id] = mean_emb

        # Сохраняем в Supabase
        try:
            from database import save_embedding
            save_embedding(user_id, mean_emb.tolist())
        except Exception as e:
            print(f"Supabase недоступен, сохраняю локально: {e}")
            self._save_embeddings()  # fallback

        return {
            'success': True,
            'user_id': user_id,
            'photos_used': len(images),
            'message': f'Пользователь {user_id} зарегистрирован по {len(images)} фото'
        }

    def verify(self, user_id: str, probe_image: Image.Image) -> dict:
        """
        Верифицирует изображение против эталона пользователя.
        Возвращает similarity score и решение.
        """
        if user_id not in self.user_embeddings:
            return {
                'verified': False,
                'similarity': 0.0,
                'threshold': self.threshold,
                'error': f'Пользователь {user_id} не зарегистрирован'
            }

        probe_emb = self.get_embedding(probe_image)
        ref_emb = self.user_embeddings[user_id].to(probe_emb.device)

        similarity = F.cosine_similarity(
            probe_emb.unsqueeze(0),
            ref_emb.unsqueeze(0)
        ).item()

        verified = similarity >= self.threshold

        return {
            'verified': verified,
            'similarity': round(similarity, 4),
            'threshold': self.threshold,
            'user_id': user_id,
            'decision': 'ACCEPTED' if verified else 'REJECTED',
            'confidence': self._get_confidence(similarity)
        }

    def verify_against_all(self, probe_image: Image.Image) -> dict:
        """
        Проверяет изображение против всех зарегистрированных пользователей.
        Полезно для идентификации (не только верификации).
        """
        if not self.user_embeddings:
            return {'found': False, 'scores': {}}

        probe_emb = self.get_embedding(probe_image)
        scores = {}

        for uid, ref_emb in self.user_embeddings.items():
            sim = F.cosine_similarity(
                probe_emb.unsqueeze(0),
                ref_emb.unsqueeze(0)
            ).item()
            scores[uid] = round(sim, 4)

        best_user = max(scores, key=scores.get)
        best_score = scores[best_user]

        return {
            'found': best_score >= self.threshold,
            'best_match': best_user if best_score >= self.threshold else None,
            'best_score': round(best_score, 4),
            'scores': scores,
            'threshold': self.threshold
        }

    def _get_confidence(self, similarity: float) -> str:
        """Возвращает текстовый уровень уверенности."""
        if similarity >= 0.90:
            return 'very_high'
        elif similarity >= 0.80:
            return 'high'
        elif similarity >= 0.70:
            return 'medium'
        elif similarity >= 0.55:
            return 'low'
        else:
            return 'very_low'

    def get_registered_users(self) -> list[str]:
        """Возвращает список зарегистрированных пользователей."""
        return list(self.user_embeddings.keys())

    def delete_user(self, user_id: str) -> bool:
        """Удаляет пользователя из Supabase и кэша памяти."""
        if user_id in self.user_embeddings:
            del self.user_embeddings[user_id]

            # Удаляем из Supabase
            try:
                from database import delete_user_embedding
                delete_user_embedding(user_id)
            except Exception as e:
                print(f"Ошибка удаления из Supabase: {e}")
                self._save_embeddings()  # fallback

            return True
        return False