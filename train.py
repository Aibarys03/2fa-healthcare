"""
Обучение CNN с Triplet Loss для Few-Shot Face Recognition
Запуск: python train.py --data_dir data/users --epochs 30
"""

import os
import json
import argparse
import random
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import FaceEmbeddingNet, TripletLoss


# ─── Конфигурация ───────────────────────────────────────────────────────────

IMG_SIZE = 100
EMBEDDING_DIM = 128
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MARGIN = 0.3
VERIFICATION_THRESHOLD = 0.70


# ─── Датасет ────────────────────────────────────────────────────────────────

class TripletFaceDataset(Dataset):
    """
    Генерирует триплеты (anchor, positive, negative) из папки с пользователями.

    Структура папки:
        data/users/
            user_1/  ← 5-10 фото
                img1.jpg
                img2.jpg
            user_2/
                img1.jpg
                ...
    """

    def __init__(self, data_dir: str, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.user_images = {}  # {user_id: [paths]}
        self.triplets = []

        self._load_images()
        self._generate_triplets()
        self._setup_transforms()

    def _load_images(self):
        """Загружает пути к изображениям по пользователям."""
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        users_found = 0

        for user_dir in sorted(self.data_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            images = [
                p for p in user_dir.iterdir()
                if p.suffix.lower() in valid_ext
            ]
            if len(images) >= 2:  # минимум 2 фото для tripler
                self.user_images[user_dir.name] = images
                users_found += 1
                print(f"  [OK] {user_dir.name}: {len(images)} фото")

        print(f"\nВсего пользователей: {users_found}")
        if users_found < 2:
            raise ValueError(
                "Нужно минимум 2 пользователя для обучения!\n"
                "Создайте папки: data/users/user_1/, data/users/user_2/, ..."
            )

    def _generate_triplets(self):
        """Генерирует триплеты: (anchor, positive, negative)."""
        users = list(self.user_images.keys())
        self.triplets = []

        for user in users:
            imgs = self.user_images[user]
            other_users = [u for u in users if u != user]

            # Для каждого изображения создаём несколько триплетов
            for anchor_path in imgs:
                # Positive: другое фото того же пользователя
                pos_candidates = [p for p in imgs if p != anchor_path]
                if not pos_candidates:
                    continue

                for _ in range(3):  # 3 триплета на anchor
                    positive_path = random.choice(pos_candidates)
                    # Negative: фото другого пользователя
                    neg_user = random.choice(other_users)
                    negative_path = random.choice(self.user_images[neg_user])
                    self.triplets.append((anchor_path, positive_path, negative_path))

        random.shuffle(self.triplets)
        print(f"Сгенерировано триплетов: {len(self.triplets)}")

    def _setup_transforms(self):
        """Трансформации для обучения и инференса."""
        self.train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        if self.augment:
            return self.train_transform(img)
        return self.val_transform(img)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]
        return (
            self._load_image(anchor_path),
            self._load_image(positive_path),
            self._load_image(negative_path)
        )


# ─── Обучение ───────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for anchor, positive, negative in dataloader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss, pos_dist, neg_dist = criterion(emb_a, emb_p, emb_n)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        # Точность: pos_distance < neg_distance
        correct += int(pos_dist < neg_dist)
        total += 1

    return total_loss / total, correct / total * 100


def evaluate(model, dataset, device, n_samples=100):
    """Оценка через cosine similarity на случайных парах."""
    model.eval()
    users = list(dataset.user_images.keys())
    if len(users) < 2:
        return 0.0

    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(n_samples):
            # Genuine pair (тот же пользователь)
            user = random.choice(users)
            imgs = dataset.user_images[user]
            if len(imgs) < 2:
                continue
            a, b = random.sample(imgs, 2)

            img_a = dataset.val_transform(Image.open(a).convert('RGB')).unsqueeze(0).to(device)
            img_b = dataset.val_transform(Image.open(b).convert('RGB')).unsqueeze(0).to(device)

            emb_a = model(img_a)
            emb_b = model(img_b)

            sim_genuine = torch.nn.functional.cosine_similarity(emb_a, emb_b).item()

            # Impostor pair (разные пользователи)
            other_users = [u for u in users if u != user]
            other_user = random.choice(other_users)
            img_c_path = random.choice(dataset.user_images[other_user])
            img_c = dataset.val_transform(Image.open(img_c_path).convert('RGB')).unsqueeze(0).to(device)
            emb_c = model(img_c)

            sim_impostor = torch.nn.functional.cosine_similarity(emb_a, emb_c).item()

            # Correct: genuine выше порога, impostor ниже
            if sim_genuine >= VERIFICATION_THRESHOLD and sim_impostor < VERIFICATION_THRESHOLD:
                correct += 1
            total += 1

    return correct / total * 100 if total > 0 else 0.0


def save_training_plot(history: dict, save_path: str):
    """Сохраняет графики обучения."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0f0f1a')

    epochs = range(1, len(history['loss']) + 1)

    # Loss
    ax1.set_facecolor('#1a1a2e')
    ax1.plot(epochs, history['loss'], color='#00d4ff', linewidth=2, marker='o', markersize=4)
    ax1.set_title('Training Loss', color='white', fontsize=13)
    ax1.set_xlabel('Epoch', color='#aaa')
    ax1.set_ylabel('Triplet Loss', color='#aaa')
    ax1.tick_params(colors='#aaa')
    ax1.grid(True, alpha=0.2, color='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333')

    # Accuracy
    ax2.set_facecolor('#1a1a2e')
    ax2.plot(epochs, history['accuracy'], color='#7c3aed', linewidth=2, marker='o', markersize=4)
    ax2.set_title('Verification Accuracy', color='white', fontsize=13)
    ax2.set_xlabel('Epoch', color='#aaa')
    ax2.set_ylabel('Accuracy (%)', color='#aaa')
    ax2.tick_params(colors='#aaa')
    ax2.grid(True, alpha=0.2, color='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()


def train(data_dir: str, epochs: int = 30, save_dir: str = 'models'):
    """Основная функция обучения."""
    print("=" * 60)
    print("  Few-Shot Face Recognition — Triplet Loss Training")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Датасет
    print(f"\nЗагрузка данных из: {data_dir}")
    dataset = TripletFaceDataset(data_dir, augment=True)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # Модель
    model = FaceEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(device)
    criterion = TripletLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {total_params:,}")

    # Обучение
    history = {'loss': [], 'accuracy': [], 'epoch_time': []}
    best_accuracy = 0.0
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nНачало обучения: {epochs} эпох\n")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Train Acc':>10} | {'Val Acc':>8} | {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        loss, train_acc = train_epoch(model, dataloader, criterion, optimizer, device)
        val_acc = evaluate(model, dataset, device, n_samples=50)
        scheduler.step()

        elapsed = time.time() - t0
        history['loss'].append(loss)
        history['accuracy'].append(val_acc)
        history['epoch_time'].append(elapsed)

        print(f"{epoch:>6} | {loss:>8.4f} | {train_acc:>9.1f}% | {val_acc:>7.1f}% | {elapsed:>5.1f}s")

        # Сохранение лучшей модели
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'loss': loss,
                'embedding_dim': EMBEDDING_DIM,
                'img_size': IMG_SIZE,
                'threshold': VERIFICATION_THRESHOLD,
                'users': list(dataset.user_images.keys())
            }, os.path.join(save_dir, 'best_model.pth'))

    # Финальное сохранение
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'accuracy': val_acc,
        'loss': loss,
        'embedding_dim': EMBEDDING_DIM,
        'img_size': IMG_SIZE,
        'threshold': VERIFICATION_THRESHOLD,
        'users': list(dataset.user_images.keys())
    }, os.path.join(save_dir, 'last_model.pth'))

    # Сохранение истории
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # График
    plot_path = os.path.join(save_dir, 'training_plot.png')
    save_training_plot(history, plot_path)

    print("\n" + "=" * 55)
    print(f"  Обучение завершено!")
    print(f"  Лучшая точность: {best_accuracy:.1f}%")
    print(f"  Модель сохранена: {save_dir}/best_model.pth")
    print(f"  График: {plot_path}")
    print("=" * 55)

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Few-Shot Face Recognition')
    parser.add_argument('--data_dir', type=str, default='data/users',
                        help='Путь к папке с пользователями')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Количество эпох обучения')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Папка для сохранения модели')
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.save_dir)
