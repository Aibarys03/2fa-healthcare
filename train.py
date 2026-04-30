"""
Обучение CNN с Triplet Loss для Few-Shot Face Recognition (v2)

Основные улучшения по сравнению с v1:

  1. Чёткий train/val split на уровне ПАР (не утечка фото между фолдами).
  2. Большая и детерминированная валидационная выборка (200 пар, фиксированный seed).
  3. Усиленные аугментации (HorizontalFlip 0.5, RandomResizedCrop, ColorJitter,
     RandomAffine для сдвигов).
  4. Semi-hard online triplet mining внутри batch (FaceNet-style).
  5. Early stopping (patience=7).
  6. Финальная развёрнутая оценка: FAR, FRR, GAR, EER, ROC AUC, средние
     similarity для genuine vs impostor.

Запуск: python train.py --data_dir data/users --epochs 30
"""

import os
import json
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
VERIFICATION_THRESHOLD = 0.80

VAL_FRACTION = 0.25       # 25% фото каждого пользователя — на валидацию
VAL_SAMPLE_PAIRS = 200    # сколько пар (genuine + impostor) на оценку
EARLY_STOP_PATIENCE = 7   # эпох без улучшения до остановки

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─── Датасет ────────────────────────────────────────────────────────────────

class TripletFaceDataset(Dataset):
    """
    Генерирует триплеты (anchor, positive, negative) из train-сплита.
    Validation-фото отдельно — используются только для evaluate().
    """

    def __init__(self, data_dir: str, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.train_images = {}   # {user_id: [paths]}  - для обучения
        self.val_images = {}     # {user_id: [paths]}  - для валидации
        self.triplets = []

        self._load_images()
        self._generate_triplets()
        self._setup_transforms()

    def _load_images(self):
        """Загружает фото и разделяет на train/val по фотографиям внутри пользователя."""
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        users_found = 0

        for user_dir in sorted(self.data_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            images = sorted([
                p for p in user_dir.iterdir()
                if p.suffix.lower() in valid_ext
            ])
            if len(images) < 4:
                # Минимум 4 фото: 3 на train, 1 на val
                print(f"  [SKIP] {user_dir.name}: only {len(images)} photos (need ≥4)")
                continue

            # Детерминированное разделение train/val
            random.Random(SEED).shuffle(images)
            n_val = max(1, int(len(images) * VAL_FRACTION))
            val_imgs = images[:n_val]
            train_imgs = images[n_val:]

            self.train_images[user_dir.name] = train_imgs
            self.val_images[user_dir.name] = val_imgs
            users_found += 1
            print(f"  [OK] {user_dir.name}: {len(images)} фото "
                  f"(train={len(train_imgs)}, val={len(val_imgs)})")

        print(f"\nВсего пользователей: {users_found}")
        if users_found < 2:
            raise ValueError(
                "Нужно минимум 2 пользователя с ≥4 фото для обучения!"
            )

    def _generate_triplets(self):
        """Генерирует триплеты ТОЛЬКО из train-фото."""
        users = list(self.train_images.keys())
        self.triplets = []

        for user in users:
            imgs = self.train_images[user]
            other_users = [u for u in users if u != user]

            for anchor_path in imgs:
                pos_candidates = [p for p in imgs if p != anchor_path]
                if not pos_candidates:
                    continue

                # 5 триплетов на anchor (было 3) — больше разнообразия
                for _ in range(5):
                    positive_path = random.choice(pos_candidates)
                    neg_user = random.choice(other_users)
                    negative_path = random.choice(self.train_images[neg_user])
                    self.triplets.append((anchor_path, positive_path, negative_path))

        random.shuffle(self.triplets)
        print(f"Сгенерировано триплетов (train): {len(self.triplets)}")

    def _setup_transforms(self):
        """Усиленные аугментации для обучения."""
        self.train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE + 12, IMG_SIZE + 12)),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                   saturation=0.15, hue=0.05),
            transforms.RandomRotation(12),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
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


# ─── Обучение с semi-hard mining ──────────────────────────────────────────

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Обучение с semi-hard mining поверх pre-generated триплетов.

    Внутри batch для каждого якоря пересортируем negatives так, чтобы выбрать
    самый сложный (но не слишком — чтобы не свалиться в тупик).
    """
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

        # Semi-hard mining внутри batch: для каждого якоря выбираем самый
        # сложный negative из всех негативов в этом же batch.
        with torch.no_grad():
            # Расстояния anchor-к-всем-negatives в батче
            dist_a_n_all = torch.cdist(emb_a, emb_n)            # [B, B]
            dist_a_p = (emb_a - emb_p).pow(2).sum(1).sqrt()    # [B]
            # Маска: выбираем negatives, которые сложнее (ближе к anchor),
            # но всё ещё дальше positive (semi-hard зона)
            mask_semihard = (dist_a_n_all > dist_a_p.unsqueeze(1)) & \
                            (dist_a_n_all < dist_a_p.unsqueeze(1) + MARGIN * 2)
            # Если в строке нет semi-hard кандидатов, оставляем оригинальный
            best_neg_idx = []
            for i in range(emb_a.size(0)):
                row = mask_semihard[i].nonzero().flatten()
                if row.numel() > 0:
                    # Берём самый ближний из semi-hard
                    distances_in_row = dist_a_n_all[i, row]
                    best_neg_idx.append(row[distances_in_row.argmin()].item())
                else:
                    best_neg_idx.append(i)
            best_neg_idx = torch.tensor(best_neg_idx, device=device)

        # Перестраиваем emb_n с учётом лучших negatives
        emb_n_mined = emb_n[best_neg_idx]

        loss, pos_dist, neg_dist = criterion(emb_a, emb_p, emb_n_mined)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += int(pos_dist < neg_dist)
        total += 1

    return total_loss / total, correct / total * 100


# ─── Детерминированная валидация ────────────────────────────────────────

class FixedEvalSet:
    """
    Фиксированный набор пар для валидации.
    Создаётся один раз и не меняется между эпохами.
    """
    def __init__(self, val_images: dict, n_pairs: int, seed: int = 1234):
        rng = random.Random(seed)
        users = list(val_images.keys())
        self.genuine = []   # [(path_a, path_b), ...]
        self.impostor = []

        n_each = n_pairs // 2

        # Genuine: пары из одного пользователя
        for _ in range(n_each):
            user = rng.choice(users)
            imgs = val_images[user]
            if len(imgs) < 2:
                continue
            a, b = rng.sample(imgs, 2)
            self.genuine.append((a, b))

        # Impostor: пары из разных пользователей
        for _ in range(n_each):
            u1, u2 = rng.sample(users, 2)
            a = rng.choice(val_images[u1])
            b = rng.choice(val_images[u2])
            self.impostor.append((a, b))

        print(f"Фиксированная валидация: {len(self.genuine)} genuine + "
              f"{len(self.impostor)} impostor пар")


def compute_similarities(model, pairs, transform, device):
    """Вычисляет cosine similarity для списка пар (path_a, path_b)."""
    model.eval()
    sims = []
    with torch.no_grad():
        for a, b in pairs:
            img_a = transform(Image.open(a).convert('RGB')).unsqueeze(0).to(device)
            img_b = transform(Image.open(b).convert('RGB')).unsqueeze(0).to(device)
            emb_a = model(img_a)
            emb_b = model(img_b)
            sim = F.cosine_similarity(emb_a, emb_b).item()
            sims.append(sim)
    return np.array(sims)


def evaluate(model, eval_set: FixedEvalSet, transform, device, threshold=0.70):
    """Полная оценка: accuracy, FAR, FRR, GAR, EER, ROC AUC."""
    sim_genuine = compute_similarities(model, eval_set.genuine, transform, device)
    sim_impostor = compute_similarities(model, eval_set.impostor, transform, device)

    # Точность по фиксированному порогу
    genuine_correct = (sim_genuine >= threshold).sum()
    impostor_correct = (sim_impostor < threshold).sum()
    total = len(sim_genuine) + len(sim_impostor)
    accuracy = (genuine_correct + impostor_correct) / total * 100

    # FAR / FRR
    far = (sim_impostor >= threshold).sum() / len(sim_impostor) * 100
    frr = (sim_genuine < threshold).sum() / len(sim_genuine) * 100
    gar = 100 - frr

    # EER — точка где FAR = FRR при переборе порогов
    # ВАЖНО: cosine similarity находится в диапазоне [-1, 1], а не [0, 1]
    thresholds = np.linspace(-1.0, 1.0, 2001)
    fars = np.array([(sim_impostor >= t).sum() / len(sim_impostor) for t in thresholds])
    frrs = np.array([(sim_genuine < t).sum() / len(sim_genuine) for t in thresholds])
    eer_idx = np.argmin(np.abs(fars - frrs))
    eer = (fars[eer_idx] + frrs[eer_idx]) / 2 * 100

    # ROC AUC — площадь под кривой TPR vs FPR
    tprs = 1 - frrs
    fprs = fars

    order = np.argsort(fprs)
    fprs_sorted = fprs[order]
    tprs_sorted = tprs[order]
    fprs_full = np.concatenate([[0.0], fprs_sorted, [1.0]])
    tprs_full = np.concatenate([[0.0], tprs_sorted, [1.0]])

    if hasattr(np, 'trapezoid'):
        auc = np.trapezoid(tprs_full, fprs_full)
    else:
        auc = np.trapz(tprs_full, fprs_full)

    # Поиск оптимального порога — максимизируем (1 - FAR) + (1 - FRR)
    # = TNR + TPR = Youden's J statistic + 1
    youden_j = (1 - fars) + (1 - frrs) - 1
    best_thr_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_thr_idx]
    best_far = fars[best_thr_idx] * 100
    best_frr = frrs[best_thr_idx] * 100

    return {
        'accuracy': accuracy,
        'far': far,
        'frr': frr,
        'gar': gar,
        'eer': eer,
        'roc_auc': auc,
        'mean_genuine': float(sim_genuine.mean()),
        'mean_impostor': float(sim_impostor.mean()),
        'std_genuine': float(sim_genuine.std()),
        'std_impostor': float(sim_impostor.std()),
        'optimal_threshold': float(best_threshold),
        'optimal_far': float(best_far),
        'optimal_frr': float(best_frr),
    }


def save_training_plot(history: dict, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0f0f1a')
    epochs = range(1, len(history['loss']) + 1)

    ax1.set_facecolor('#1a1a2e')
    ax1.plot(epochs, history['loss'], color='#00d4ff', linewidth=2, marker='o', markersize=4)
    ax1.set_title('Training Loss', color='white', fontsize=13)
    ax1.set_xlabel('Epoch', color='#aaa')
    ax1.set_ylabel('Triplet Loss', color='#aaa')
    ax1.tick_params(colors='#aaa')
    ax1.grid(True, alpha=0.2, color='white')
    for spine in ax1.spines.values(): spine.set_edgecolor('#333')

    ax2.set_facecolor('#1a1a2e')
    ax2.plot(epochs, history['accuracy'], color='#7c3aed', linewidth=2,
             marker='o', markersize=4, label='Val Accuracy')
    ax2.plot(epochs, history['train_accuracy'], color='#10b981', linewidth=2,
             marker='s', markersize=4, label='Train Accuracy', alpha=0.7)
    ax2.set_title('Verification Accuracy', color='white', fontsize=13)
    ax2.set_xlabel('Epoch', color='#aaa')
    ax2.set_ylabel('Accuracy (%)', color='#aaa')
    ax2.tick_params(colors='#aaa')
    ax2.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax2.grid(True, alpha=0.2, color='white')
    for spine in ax2.spines.values(): spine.set_edgecolor('#333')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()


# ─── Основная функция ───────────────────────────────────────────────────

def train(data_dir: str, epochs: int = 30, save_dir: str = 'models'):
    print("=" * 60)
    print("  Few-Shot Face Recognition — Triplet Loss Training (v2)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nЗагрузка данных из: {data_dir}")
    dataset = TripletFaceDataset(data_dir, augment=True)

    # Фиксированный набор валидации
    eval_set = FixedEvalSet(dataset.val_images, n_pairs=VAL_SAMPLE_PAIRS, seed=1234)

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True
    )

    model = FaceEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(device)
    criterion = TripletLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {total_params:,}")

    history = {'loss': [], 'accuracy': [], 'train_accuracy': [], 'epoch_time': []}
    best_accuracy = 0.0
    epochs_no_improve = 0
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nНачало обучения: до {epochs} эпох (early stop patience={EARLY_STOP_PATIENCE})\n")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Train':>7} | {'Val':>7} | {'EER':>6} | {'AUC':>6} | {'Time':>6}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        loss, train_acc = train_epoch(model, dataloader, criterion, optimizer, device)
        metrics = evaluate(model, eval_set, dataset.val_transform, device,
                          threshold=VERIFICATION_THRESHOLD)
        scheduler.step()

        elapsed = time.time() - t0
        val_acc = metrics['accuracy']
        history['loss'].append(loss)
        history['accuracy'].append(val_acc)
        history['train_accuracy'].append(train_acc)
        history['epoch_time'].append(elapsed)

        marker = ""
        if val_acc > best_accuracy:
            marker = " *"
            best_accuracy = val_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # Старые поля — для совместимости с verifier.py / app.py
                'accuracy': float(val_acc),
                'loss': float(loss),
                'embedding_dim': EMBEDDING_DIM,
                'img_size': IMG_SIZE,
                'threshold': float(VERIFICATION_THRESHOLD),
                'users': list(dataset.train_images.keys()),
                # Новые расширенные метрики
                'metrics': metrics,
            }, os.path.join(save_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1

        print(f"{epoch:>6} | {loss:>8.4f} | {train_acc:>6.1f}% | "
              f"{val_acc:>6.1f}% | {metrics['eer']:>5.1f}% | "
              f"{metrics['roc_auc']:>5.3f} | {elapsed:>5.1f}s{marker}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n  ⏹  Early stopping: {EARLY_STOP_PATIENCE} эпох без улучшения")
            break

    # Финал — переоценка лучшей модели
    print("\n" + "=" * 70)
    print("  Финальная оценка лучшей модели:")
    best_ckpt = torch.load(os.path.join(save_dir, 'best_model.pth'),
                          map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    final_metrics = evaluate(model, eval_set, dataset.val_transform, device,
                            threshold=VERIFICATION_THRESHOLD)

    print(f"  Сохранена эпоха:        {best_ckpt['epoch']}")
    print(f"  Verification Accuracy:  {final_metrics['accuracy']:.1f}%")
    print(f"  FAR  (False Accept):    {final_metrics['far']:.2f}%")
    print(f"  FRR  (False Reject):    {final_metrics['frr']:.2f}%")
    print(f"  GAR  (Genuine Accept):  {final_metrics['gar']:.2f}%")
    print(f"  EER  (Equal Error):     {final_metrics['eer']:.2f}%")
    print(f"  ROC AUC:                {final_metrics['roc_auc']:.4f}")
    print(f"  Mean genuine sim:       {final_metrics['mean_genuine']:.3f} ± "
          f"{final_metrics['std_genuine']:.3f}")
    print(f"  Mean impostor sim:      {final_metrics['mean_impostor']:.3f} ± "
          f"{final_metrics['std_impostor']:.3f}")
    print(f"  Separation margin:      "
          f"{final_metrics['mean_genuine'] - final_metrics['mean_impostor']:.3f}")
    print(f"\n  ── Оптимальный порог (по Youden's J) ──")
    print(f"  Optimal threshold:      {final_metrics['optimal_threshold']:.3f}")
    print(f"  При этом пороге FAR:    {final_metrics['optimal_far']:.2f}%")
    print(f"  При этом пороге FRR:    {final_metrics['optimal_frr']:.2f}%")
    if final_metrics['optimal_threshold'] != VERIFICATION_THRESHOLD:
        print(f"\n  ⚠ Текущий порог в коде: {VERIFICATION_THRESHOLD:.2f}")
        print(f"  Рекомендуется обновить VERIFICATION_THRESHOLD в verifier.py")
        print(f"  на {final_metrics['optimal_threshold']:.2f} для лучшего баланса.")

    # Сохранение
    # Формат training_history.json — обратно-совместимый со старой версией
    # (на верхнем уровне поля loss/accuracy/epoch_time как раньше),
    # с дополнительными полями final_metrics, best_epoch, config
    history_to_save = {
        'loss': history['loss'],
        'accuracy': history['accuracy'],
        'train_accuracy': history['train_accuracy'],
        'epoch_time': history['epoch_time'],
        'final_metrics': final_metrics,
        'best_epoch': best_ckpt['epoch'],
        'config': {
            'epochs_run': len(history['loss']),
            'epochs_max': epochs,
            'batch_size': BATCH_SIZE,
            'lr': LEARNING_RATE,
            'margin': MARGIN,
            'threshold': VERIFICATION_THRESHOLD,
            'val_fraction': VAL_FRACTION,
            'val_pairs': VAL_SAMPLE_PAIRS,
        }
    }
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history_to_save, f, indent=2)

    plot_path = os.path.join(save_dir, 'training_plot.png')
    save_training_plot(history, plot_path)

    print(f"\n  Модель:    {save_dir}/best_model.pth")
    print(f"  График:    {plot_path}")
    print(f"  Метрики:   {save_dir}/training_history.json")
    print("=" * 70)

    return history, final_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Few-Shot Face Recognition')
    parser.add_argument('--data_dir', type=str, default='data/users')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.save_dir)
