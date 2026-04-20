#!/usr/bin/env python3"""
#Вспомогательный скрипт: копирует training_plot.png в static/ при каждом запросе.
#Запускается автоматически из app.py через middleware.
"""

import shutil
import os
from pathlib import Path


def sync_static():
    """Синхронизирует файлы из models/ в static/."""
    os.makedirs('static', exist_ok=True)
    
    # Копируем график обучения
    plot_src = Path('models/training_plot.png')
    if plot_src.exists():
        shutil.copy2(plot_src, 'static/training_plot.png')


if __name__ == '__main__':
    sync_static()
    print("Static files synced.")
