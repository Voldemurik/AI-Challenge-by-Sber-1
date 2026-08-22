"""Загрузка и проверка данных конкурса."""

from pathlib import Path

import numpy as np
import pandas as pd

# Корень проекта: src/data.py -> src -> корень
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# В архиве конкурса файлы лежат без расширений. Поддерживаем оба варианта.
FILENAMES = {
    "train": ("train", "train.csv"),
    "test": ("test", "test.csv"),
    "sample_submission": ("sample_submission", "sample_submission.csv"),
}

PRICE_FLOOR = 350.0  # в train арендная ставка обрезана снизу этим значением
TARGET = "rent_price"


def _find(kind: str) -> Path:
    for name in FILENAMES[kind]:
        path = DATA_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Не найден файл '{kind}' в {DATA_DIR}. "
        f"Положите туда train, test и sample_submission из архива конкурса."
    )


def load(kind: str) -> pd.DataFrame:
    """Прочитать один файл: 'train', 'test' или 'sample_submission'."""
    return pd.read_csv(_find(kind))


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Прочитать train, test и sample_submission одним вызовом."""
    return load("train"), load("test"), load("sample_submission")


def sanity_check(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Напечатать проверки, от которых зависит, можно ли доверять валидации."""
    print(f"train: {train.shape}   test: {test.shape}")
    print(f"Пропусков: train {train.isna().sum().sum()}, test {test.isna().sum().sum()}")
    print(f"Дубликатов в train (без id): {train.drop(columns='id').duplicated().sum()}")
    print(f"Корреляция id с таргетом: {np.corrcoef(train.id, train[TARGET])[0, 1]:+.4f}")

    print("\nСдвиг распределений train vs test:")
    for col in ("area", "floor", "distance_to_center"):
        print(f"  {col:20s} train {train[col].mean():7.2f}   test {test[col].mean():7.2f}")

    shift = (test.district.value_counts(normalize=True)
             - train.district.value_counts(normalize=True)) * 100
    print(f"\nМаксимальное расхождение долей районов: {shift.abs().max():.2f} п.п.")
