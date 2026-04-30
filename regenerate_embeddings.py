"""
Пересчитывает embeddings для всех пользователей в data/users/
с использованием ТЕКУЩЕЙ модели и сохраняет в Supabase.

Запускать после каждого переобучения CNN!

Использование:
  python regenerate_embeddings.py

Опции:
  --data_dir <path>   Папка с пользователями (default: data/users)
  --average           Брать среднее по всем фото пользователя (рекомендуется)
                      вместо одного embedding с первого фото
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import FaceEmbeddingNet
from database import save_embedding, get_all_users  # ← ваш database.py

IMG_SIZE = 100
EMBEDDING_DIM = 128


def load_model(checkpoint_path: str, device):
    """Загружает обученную CNN."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FaceEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Модель загружена: {checkpoint_path}")
    print(f"  Эпоха: {ckpt.get('epoch', '?')}")
    print(f"  Accuracy: {ckpt.get('accuracy', '?')}%")
    print(f"  Threshold: {ckpt.get('threshold', '?')}")
    return model


def get_transform():
    """Тот же transform что используется при инференсе."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def compute_embedding(model, transform, img_path: Path, device) -> np.ndarray:
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).cpu().numpy().flatten()
    return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/users')
    parser.add_argument('--model', default='models/best_model.pth')
    parser.add_argument('--average', action='store_true',
                        help='Усреднить embedding по всем фото (рекомендуется)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    model = load_model(args.model, device)
    transform = get_transform()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"✗ Папка {data_dir} не найдена")
        sys.exit(1)

    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    users_processed = 0
    users_failed = 0

    print(f"\nОбрабатываю пользователей из {data_dir}:\n")

    for user_dir in sorted(data_dir.iterdir()):
        if not user_dir.is_dir():
            continue

        images = [
            p for p in sorted(user_dir.iterdir())
            if p.suffix.lower() in valid_ext
        ]
        if not images:
            print(f"  [SKIP] {user_dir.name}: нет изображений")
            continue

        try:
            if args.average:
                # Усреднение по всем фото — даёт более устойчивый embedding
                embeddings = []
                for img_path in images:
                    emb = compute_embedding(model, transform, img_path, device)
                    embeddings.append(emb)
                final_emb = np.mean(embeddings, axis=0)
                # L2-normalize
                final_emb = final_emb / (np.linalg.norm(final_emb) + 1e-8)
                used = f"среднее по {len(images)} фото"
            else:
                # Использовать только первое фото
                final_emb = compute_embedding(model, transform, images[0], device)
                used = f"первое фото из {len(images)}"

            # Сохраняем в Supabase
            save_embedding(user_dir.name, final_emb.tolist())
            print(f"  [OK] {user_dir.name}: {used}, ||emb||={np.linalg.norm(final_emb):.3f}")
            users_processed += 1
        except Exception as e:
            print(f"  [FAIL] {user_dir.name}: {e}")
            users_failed += 1

    print(f"\n{'='*50}")
    print(f"  Обработано пользователей: {users_processed}")
    print(f"  Ошибок:                   {users_failed}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
