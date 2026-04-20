"""
Lightweight CNN with Triplet Loss for Few-Shot Face Recognition
Специально оптимизирован для малых данных (5-10 фото на пользователя)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceEmbeddingNet(nn.Module):
    """
    Лёгкая CNN для генерации 128-мерных эмбеддингов лиц.
    Намеренно упрощена для CPU-инференса и few-shot обучения.
    """

    def __init__(self, embedding_dim=128):
        super(FaceEmbeddingNet, self).__init__()

        # Блок 1: базовые признаки
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 100x100 -> 50x50
            nn.Dropout2d(0.1)
        )

        # Блок 2: средние признаки
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 50x50 -> 25x25
            nn.Dropout2d(0.1)
        )

        # Блок 3: высокоуровневые признаки
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 25x25 -> 12x12
            nn.Dropout2d(0.15)
        )

        # Блок 4: компактное представление
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # любой размер -> 4x4
        )

        # Полносвязные слои
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # L2-нормализация для косинусного сходства
        x = F.normalize(x, p=2, dim=1)
        return x


class TripletLoss(nn.Module):
    """
    Triplet Loss: обучает модель сближать anchor-positive
    и отдалять anchor-negative эмбеддинги.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Косинусное расстояние (1 - similarity)
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)

        # Triplet loss с margin
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean(), pos_dist.mean().item(), neg_dist.mean().item()


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Вычисляет косинусное сходство между двумя эмбеддингами."""
    if emb1.dim() == 1:
        emb1 = emb1.unsqueeze(0)
    if emb2.dim() == 1:
        emb2 = emb2.unsqueeze(0)
    sim = F.cosine_similarity(emb1, emb2)
    return sim.item()
